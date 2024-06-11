import argparse
import os
import sys
import numpy as np
import torch as th
from torchvision.utils import save_image, make_grid
from torchvision.transforms import ToTensor
from diffusers import StableDiffusionPipeline
from tqdm import tqdm
from PIL import Image
import lpips
from sklearn.model_selection import train_test_split
from accelerate import Accelerator

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, project_root)

def read_csv_with_commas(file_path, occupation):
    with open(file_path, "r") as f:
        lines = f.readlines()

    prompts = []
    comma_count = occupation.count(",")
    for line in lines:
        parts = line.strip().split(",")
        if comma_count == 0:
            image_id, normal_prompt, sensitive_prompt = parts
        else:
            image_id = parts[0]
            normal_prompt = ",".join(parts[1:comma_count+2]).strip()
            sensitive_prompt = ",".join(parts[comma_count+2:]).strip()
        prompts.append((normal_prompt, sensitive_prompt))

    return prompts

class OccupationDataset(th.utils.data.Dataset):
    def __init__(self, data_dir, num_occupations=None):
        self.data_dir = os.path.join(script_dir, data_dir)
        if num_occupations:
            self.occupations = sorted([d for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d))])[:num_occupations]
        else:
            self.occupations = sorted([d for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d))])
        self.image_files = []
        self.prompts = []
        
        for idx, occupation in enumerate(self.occupations):
            occ_dir = os.path.join(self.data_dir, occupation)
            image_files = sorted(f for f in os.listdir(occ_dir) if f.endswith(".png"))
            self.image_files.extend(os.path.join(occ_dir, f) for f in image_files)
            
            prompts = read_csv_with_commas(os.path.join(occ_dir, "prompts.csv"), occupation)
            self.prompts.extend(prompts)
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        prompt, _ = self.prompts[idx]
        image = Image.open(image_file).convert("RGB")
        return image, prompt

def collate_fn(batch):
    images = [item[0] for item in batch]
    prompts = [item[1] for item in batch]
    images = th.stack([ToTensor()(image) for image in images])
    return images, prompts


def compute_minority_scores(images, prompts, pipe, loss_fn, args):
    batch_losses = th.zeros(len(images), args.n_iter, device=images.device)

    latents = pipe.vae.encode(images).latent_dist.sample().detach()
    latents = latents * pipe.vae.config.scaling_factor

    for i in range(args.n_iter):
        timestep = int(0.6 * pipe.scheduler.config.num_train_timesteps)
        timesteps = th.tensor([timestep] * len(images), dtype=th.long, device=images.device)
        noise = th.randn_like(latents)
        noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

        denoised_images = pipe(prompts, latents=noisy_latents).images
        denoised_images = th.stack([ToTensor()(img) for img in denoised_images]).to(images.device)

        LPIPS_loss = loss_fn(images, denoised_images)
        batch_losses[:, i] = LPIPS_loss.view(-1)

    return batch_losses.mean(dim=1)

def main():
    args = create_argparser().parse_args()

    accelerator = Accelerator()
    print(f"Using {accelerator.num_processes} GPUs.")

    loss_fn = lpips.LPIPS(net='vgg')

    print("Loading Stable Diffusion model...")
    model_id = args.model_id
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=th.float16 if args.use_fp16 else th.float32)

    print("Creating dataset...")
    dataset = OccupationDataset(args.data_dir, num_occupations=args.num_occupations)
    dataloader = th.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    pipe, loss_fn, dataloader = accelerator.prepare(pipe, loss_fn, dataloader)

    all_images = []
    all_prompts = []
    all_scores = []

    print("Computing minority scores...")
    for images, prompts in tqdm(dataloader, disable=not accelerator.is_local_main_process):
        scores = compute_minority_scores(images, prompts, pipe, loss_fn, args)
        all_images.append(images.cpu())
        all_prompts.extend(prompts)
        all_scores.append(scores.cpu())

    all_images = th.cat(all_images)
    all_scores = th.cat(all_scores)

    print("Splitting data into train and val sets...")
    train_indices, val_indices = train_test_split(range(len(all_images)), test_size=args.val_ratio)

    output_path_train = os.path.join(script_dir, args.output_dir, "train")
    output_path_val = os.path.join(script_dir, args.output_dir, "val")
    os.makedirs(output_path_train, exist_ok=True)
    os.makedirs(output_path_val, exist_ok=True)

    print("Saving data...")
    for index in tqdm(train_indices, disable=not accelerator.is_local_main_process):
        image = all_images[index]
        prompt = all_prompts[index]
        score = all_scores[index]
        image = (image * 0.5 + 0.5).clamp_(0.0, 1.0)
        save_image(image, os.path.join(output_path_train, f'{index}.png'))
        with open(os.path.join(output_path_train, f'{index}.txt'), 'w') as f:
            f.write(f"Prompt: {prompt}\nMinority Score: {score.item():.4f}")

    for index in tqdm(val_indices, disable=not accelerator.is_local_main_process):
        image = all_images[index]
        prompt = all_prompts[index]
        score = all_scores[index]
        image = (image * 0.5 + 0.5).clamp_(0.0, 1.0)
        save_image(image, os.path.join(output_path_val, f'{index}.png'))
        with open(os.path.join(output_path_val, f'{index}.txt'), 'w') as f:
            f.write(f"Prompt: {prompt}\nMinority Score: {score.item():.4f}")

    print("Dataset construction complete")

def create_argparser():
    defaults = dict(
        batch_size=25,
        use_fp16=True,
        data_dir="dataset_2",
        output_dir="dataset_2_ms",
        model_id="SG161222/Realistic_Vision_V2.0",
        val_ratio=0.05,
        n_iter=5,
        num_occupations=1
    )
    parser = argparse.ArgumentParser()
    for k, v in defaults.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    return parser

if __name__ == "__main__":
    main()