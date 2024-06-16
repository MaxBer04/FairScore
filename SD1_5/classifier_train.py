import argparse
import os
import torch as th
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from accelerate import Accelerator
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    add_dict_to_argparser,
    args_to_dict,
    classifier_and_diffusion_defaults,
    classifier_defaults,
    create_classifier,
)
from utils.dataset import MinorityScoreDataset
from utils.custom_pipe import CustomDiffusionPipeline
import wandb

script_dir = os.path.dirname(os.path.abspath(__file__))

def main():
    args = create_argparser().parse_args()
    args.data_dir = os.path.join(script_dir, args.data_dir)
    args.resume_from_checkpoint = os.path.join(script_dir, args.resume_from_checkpoint)

    accelerator = Accelerator(mixed_precision="fp16" if args.use_fp16 else "no")

    if accelerator.is_main_process:
        logger.configure()

    if accelerator.is_main_process:
        logger.log("creating model...")
    model = create_classifier(**args_to_dict(args, classifier_defaults(out_channels=args.num_quantiles, in_channels=4).keys()))
    
    if args.resume_from_checkpoint:
        if accelerator.is_main_process:
            logger.log(f"loading model from checkpoint: {args.resume_from_checkpoint}...")
        model.load_state_dict(th.load(args.resume_from_checkpoint, map_location="cpu"))
    
    model.to(accelerator.device)

    if accelerator.is_main_process:
        logger.log("loading diffusion model...")
    diffusion_model = CustomDiffusionPipeline.from_pretrained(args.model_id)
    diffusion_model.to("cpu")
    diffusion_model.vae.to(accelerator.device)
    if args.use_fp16:
        diffusion_model.vae.to(th.float16)

    if args.feature_extractor:
        if accelerator.is_main_process:
            logger.log("loading feature extractor...")
        f_extractor = create_classifier(**args_to_dict(args, classifier_and_diffusion_defaults().keys()))
        f_extractor.load_state_dict(th.load(args.feature_extractor, map_location="cpu"))
        f_extractor.to(accelerator.device)
        if args.use_fp16:
            f_extractor.to(th.float16)
        f_extractor.eval()
        for param in f_extractor.parameters():
            param.requires_grad = False
    else:
        f_extractor = None

    if accelerator.is_main_process:
        logger.log(f"creating data loader...")

    dataset = MinorityScoreDataset(args.data_dir, args.num_quantiles)
    train_size = int(args.train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    model, train_loader, val_loader, opt = accelerator.prepare(
        model, train_loader, val_loader, th.optim.Adam(model.parameters(), lr=args.lr)
    )

    if accelerator.is_main_process:
        wandb.init(project=args.wandb_project, name=args.wandb_name, config=vars(args))

    if accelerator.is_main_process:
        logger.log("training classifier model...")

    def forward_backward_log(data_loader, prefix="train"):
        for batch in data_loader:
            with th.set_grad_enabled(prefix == "train"):
                images, quantiles = batch
                images = images.to(accelerator.device)
                quantiles = quantiles.to(accelerator.device)

                latents = diffusion_model.vae.encode(images).latent_dist.sample().detach()
                latents = latents * diffusion_model.vae.config.scaling_factor

                if args.noised:
                    t = th.randint(0, diffusion_model.scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device)
                    noise = th.randn_like(latents)
                    latents_noised = diffusion_model.scheduler.add_noise(latents, noise, t)
                else:
                    t = th.zeros(images.shape[0], dtype=th.long, device=accelerator.device)
                    latents_noised = latents

                if f_extractor is not None:
                    latents_noised = f_extractor(latents_noised)

                logits = model(latents_noised, timesteps=t)
                loss = F.cross_entropy(logits, quantiles)

                acc = (logits.argmax(dim=-1) == quantiles).float().mean()

                if accelerator.is_main_process:
                    wandb.log({f"{prefix}_loss": loss.item(), f"{prefix}_acc": acc.item()})

                if prefix == "train":
                    accelerator.backward(loss)
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    opt.zero_grad()

    start_epoch = 0
    if args.resume_from_checkpoint:
        start_epoch = int(args.resume_from_checkpoint.split("_")[-1].split(".")[0])

    for epoch in range(start_epoch, args.epochs):
        if accelerator.is_main_process:
            logger.log(f"Epoch {epoch}")
        model.train()
        forward_backward_log(train_loader)

        if accelerator.is_main_process:
            logger.log("validating...")
        model.eval()
        with th.no_grad():
            forward_backward_log(val_loader, prefix="val")

        if (epoch + 1) % args.save_interval == 0 and accelerator.is_main_process:
            logger.log(f"saving model at epoch {epoch}...")
            accelerator.save(model.state_dict(), os.path.join(wandb.run.dir, f"model_{epoch}.pt"))

    if accelerator.is_main_process:
        wandb.finish()

def create_argparser():
    defaults = dict(
        data_dir="dataset",
        lr=1e-4,
        batch_size=24,
        epochs=80,
        num_quantiles=4,
        model_id="SG161222/Realistic_Vision_V2.0",
        use_fp16=False,
        feature_extractor=None,
        noised=True,
        save_interval=4,
        train_split=0.9,
        wandb_project="minority-score-classifier",
        wandb_name="FF-08T-5it",
        resume_from_checkpoint="model_75.pt",
    )
    defaults.update(classifier_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()