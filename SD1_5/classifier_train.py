import argparse
import os
import torch as th
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from guided_diffusion import dist_util, logger
from guided_diffusion.fp16_util import MixedPrecisionTrainer
from guided_diffusion.script_util import (
    add_dict_to_argparser,
    args_to_dict,
    classifier_and_diffusion_defaults,
    classifier_defaults,
    create_classifier,
)
from utils.dataset import MinorityScoreDataset
from utils.custom_pipe import CustomDiffusionPipeline

script_dir = os.path.dirname(os.path.abspath(__file__))

def main():
    args = create_argparser().parse_args()
    args.data_dir = os.path.join(script_dir, args.data_dir)
    
    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model...")
    model = create_classifier(**args_to_dict(args, classifier_defaults(out_channels=args.num_quantiles, in_channels=4).keys()))
    model.to(dist_util.dev())

    if args.noised:
        logger.log("loading diffusion model...")
        diffusion_model = CustomDiffusionPipeline.from_pretrained(args.model_id)
        diffusion_model.to('cpu')
        diffusion_model.vae.to(dist_util.dev())
        if args.use_fp16:
            diffusion_model.vae.to(th.float16)

    if args.feature_extractor:
        logger.log("loading feature extractor...")
        f_extractor = create_classifier(**args_to_dict(args, classifier_and_diffusion_defaults().keys()))
        f_extractor.load_state_dict(dist_util.load_state_dict(args.feature_extractor, map_location="cpu"))
        f_extractor.to(dist_util.dev())
        if args.use_fp16:
            f_extractor.convert_to_fp16()
        f_extractor.eval()
        for param in f_extractor.parameters():
            param.requires_grad = False
    else:
        f_extractor = None

    logger.log(f"creating data loader...")
    
    dataset = MinorityScoreDataset(args.data_dir, args.num_quantiles)
    train_size = int(args.train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    logger.log("training classifier model...")

    def forward_backward_log(data_loader, prefix="train"):
        for batch in data_loader:
            images, quantiles = batch
            images = images.to(dist_util.dev())
            quantiles = quantiles.to(dist_util.dev())

            latents = diffusion_model.vae.encode(images).latent_dist.sample().detach()
            latents = latents * diffusion_model.vae.config.scaling_factor

            if args.noised:
                t = th.randint(0, diffusion_model.scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device)
                noise = th.randn_like(latents)
                latents_noised = diffusion_model.scheduler.add_noise(latents, noise, t)
            else:
                t = th.zeros(batch.shape[0], dtype=th.long, device=dist_util.dev())
                latents_noised = latents

            if f_extractor is not None:
                latents_noised = f_extractor(latents_noised)

            logits = model(latents_noised, timesteps=t)
            loss = F.cross_entropy(logits, quantiles)

            with th.no_grad():
                acc = (logits.argmax(dim=-1) == quantiles).float().mean()

            logger.log(f"{prefix}_loss", loss.item(), args.batch_size)
            logger.log(f"{prefix}_acc", acc.item(), args.batch_size)

            if prefix == "train":
                loss.backward()
                opt.step()
                opt.zero_grad()

    opt = th.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        logger.log(f"Epoch {epoch}")
        model.train()
        forward_backward_log(train_loader)

        logger.log("validating...")
        model.eval()
        with th.no_grad():
            forward_backward_log(val_loader, prefix="val")

        if (epoch + 1) % args.save_interval == 0:
            logger.log(f"saving model at epoch {epoch}...")
            th.save(model.state_dict(), os.path.join(logger.get_dir(), f"model_{epoch}.pt"))

def create_argparser():
    defaults = dict(
        data_dir="output-FF-08T-5it",
        lr=1e-4,
        batch_size=20,
        epochs=100,
        num_quantiles=2,
        model_id="SG161222/Realistic_Vision_V2.0",
        use_fp16=False,
        feature_extractor=None,
        noised=True,
        save_interval=10,
        train_split=0.9,
    )
    defaults.update(classifier_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()