import argparse
import os
import torch as th
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from accelerate import Accelerator
from guided_diffusion import logger
from guided_diffusion.script_util import (
    add_dict_to_argparser,
    classifier_and_diffusion_defaults,
)
from utils.dataset import HVectsDataset
from utils.classifier import make_model
import wandb
from tqdm import tqdm

script_dir = os.path.dirname(os.path.abspath(__file__))

def main():
    args = create_argparser().parse_args()
    args.data_dir = os.path.join(script_dir, args.data_dir)
    if args.resume_from_checkpoint:
        args.resume_from_checkpoint = os.path.join(script_dir, args.resume_from_checkpoint)

    accelerator = Accelerator(mixed_precision="fp16" if args.use_fp16 else "no")

    if accelerator.is_main_process:
        logger.configure()

    if accelerator.is_main_process:
        logger.log("creating model...")
        
    model = make_model(
        in_channels=args.in_channels,
        image_size=args.latents_size,
        out_channels=args.out_channels,
        combine_vectors=args.combine_vectors
    )
    
    if args.resume_from_checkpoint:
        if accelerator.is_main_process:
            logger.log(f"loading model from checkpoint: {args.resume_from_checkpoint}...")
        model.load_state_dict(th.load(args.resume_from_checkpoint, map_location="cpu"))
    
    model.to(accelerator.device)

    if accelerator.is_main_process:
        logger.log(f"creating data loader...")

    dataset = HVectsDataset(args.data_dir)
    train_size = int(args.train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    model, train_loader, val_loader, opt = accelerator.prepare(
        model, train_loader, val_loader, th.optim.Adam(model.parameters(), lr=args.lr)
    )

    if accelerator.is_main_process:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config=vars(args),
            reinit=True,
            settings=wandb.Settings(start_method="fork")
        )
        # Define custom x-axis metric
        wandb.define_metric(f"train_samples_processed", step_metric="train_progress")
        wandb.define_metric(f"val_samples_processed", step_metric="val_progress")

    if accelerator.is_main_process:
        logger.log("training classifier model...")

    def forward_backward_log(data_loader, prefix="train"):
        total_loss = 0
        total_acc = 0
        num_batches = 0

        # Get the total number of samples across all processes
        total_samples = len(data_loader.dataset)
        samples_seen = 0

        progress_bar = tqdm(total=total_samples, disable=not accelerator.is_main_process)

        for batch_idx, batch in enumerate(data_loader):
            with th.set_grad_enabled(prefix == "train"):
                h_vect = batch['h_vect']
                timesteps = batch['timestep']
                gender_scores = batch['gender_scores']

                logits = model(h_vect, timesteps)
                loss = F.cross_entropy(logits, gender_scores.argmax(dim=-1))

                # Accuracy calculation
                predicted_gender = logits.argmax(dim=-1)
                true_gender = gender_scores.argmax(dim=-1)
                acc = (predicted_gender == true_gender).float().mean()

                total_loss += loss.item()
                total_acc += acc.item()
                num_batches += 1

                if prefix == "train":
                    accelerator.backward(loss)
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    opt.zero_grad()

            # Update progress bar
            samples_in_batch = h_vect.shape[0]
            samples_seen += samples_in_batch * accelerator.num_processes
            progress_bar.update(samples_in_batch * accelerator.num_processes)

            # Log to wandb more frequently
            if accelerator.is_main_process and batch_idx % 10 == 0:  # Log every 10 batches
                wandb.log({
                    f"{prefix}_batch_loss": loss.item(),
                    f"{prefix}_batch_acc": acc.item(),
                    f"{prefix}_samples_processed": samples_seen,
                    f"{prefix}_progress": samples_seen / total_samples
                })

        progress_bar.close()

        avg_loss = total_loss / num_batches
        avg_acc = total_acc / num_batches

        if accelerator.is_main_process:
            wandb.log({
                f"{prefix}_epoch_loss": avg_loss,
                f"{prefix}_epoch_acc": avg_acc,
                f"{prefix}_samples_processed": samples_seen,
                f"{prefix}_progress": 1.0
            })

        return avg_loss, avg_acc

    start_epoch = 0
    if args.resume_from_checkpoint:
        start_epoch = int(args.resume_from_checkpoint.split("_")[-1].split(".")[0])

    for epoch in range(start_epoch, args.epochs):
        if accelerator.is_main_process:
            logger.log(f"Epoch {epoch}")
        model.train()
        train_loss, train_acc = forward_backward_log(train_loader)

        if accelerator.is_main_process:
            logger.log("validating...")
        model.eval()
        with th.no_grad():
            val_loss, val_acc = forward_backward_log(val_loader, prefix="val")

        if accelerator.is_main_process:
            logger.log(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if (epoch + 1) % args.save_interval == 0 and accelerator.is_main_process:
            logger.log(f"saving model at epoch {epoch}...")
            accelerator.save(model.state_dict(), os.path.join(wandb.run.dir, f"model_{epoch}.pt"))

    if accelerator.is_main_process:
        wandb.finish()

def create_argparser():
    defaults = classifier_and_diffusion_defaults()
    defaults.update(dict(
        data_dir="output",
        lr=1e-5,
        batch_size=32768,
        epochs=4,
        latents_size=8,
        out_channels=2,
        in_channels=2560,
        use_fp16=False,
        save_interval=4,
        train_split=0.9,
        wandb_project="h-vects-gender-classifier",
        wandb_name="hvects-gender-classifier",
        resume_from_checkpoint=None,
        combine_vectors=False,
    ))
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()