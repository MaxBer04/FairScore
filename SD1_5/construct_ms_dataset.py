import argparse
import os
import sys
import csv
import numpy as np
import torch as th
from torchvision.utils import save_image, make_grid
from torchvision.transforms import ToTensor
from diffusers import StableDiffusionPipeline
from tqdm import tqdm
from PIL import Image
import lpips
from accelerate import Accelerator
from accelerate.utils import gather_object

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
        self.occupations = [d for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d))][:num_occupations]
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
    
    def select(self, indices):
        selected_dataset = OccupationDataset(self.data_dir)
        selected_dataset.image_files = [self.image_files[i] for i in indices]
        selected_dataset.prompts = [self.prompts[i] for i in indices]
        return selected_dataset

def collate_fn(batch):
    images = [item[0] for item in batch]
    prompts = [item[1] for item in batch]
    images = th.stack([ToTensor()(image) for image in images])
    return images, prompts


def compute_minority_scores(args, dataset, pipe, loss_fn, accelerator):
    ms_tuples = []

    dataloader = th.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    
    with th.no_grad():
        for batch_idx, (images, prompts) in enumerate(tqdm(dataloader)):
            print(f"Process: {accelerator.process_index}, {prompts[0]}")
            # Konvertiere die Bilder in den richtigen Datentyp
            images = images.to(accelerator.device, dtype=pipe.vae.encoder.conv_in.bias.dtype)

            batch_losses = th.zeros(len(images), args.n_iter, device=accelerator.device)
            reconstructed_images = th.zeros(len(images) * (args.n_iter + 1), 3, images.shape[-2], images.shape[-1], device=accelerator.device)
            reconstructed_images[:len(images)] = images

            latents = pipe.vae.encode(images).latent_dist.sample().detach()
            latents = latents * pipe.vae.config.scaling_factor

            for i in range(args.n_iter):
                timestep = int(0.6 * pipe.scheduler.config.num_train_timesteps)
                timesteps = th.tensor([timestep] * len(images), dtype=th.long, device=accelerator.device)
                noise = th.randn_like(latents)
                noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

                denoised_images = pipe(prompts, latents=noisy_latents).images
                denoised_images = th.stack([ToTensor()(img) for img in denoised_images]).to(accelerator.device)
                denoised_images = (denoised_images * 2 - 1).clamp_(0.0, 1.0)
                # (image_tensor * 0.5 + 0.5).clamp_(0.0, 1.0)
                
                LPIPS_loss = loss_fn(images, denoised_images)
                batch_losses[:, i] = LPIPS_loss.view(-1)
                reconstructed_images[len(images) * (i + 1):len(images) * (i + 2)] = denoised_images

            # Speichere statt der Bilder die Indizes
            for idx, (prompt, score) in enumerate(zip(prompts, batch_losses.mean(dim=1))):
                ms_tuples.append((batch_idx * args.batch_size + idx, prompt, score))

            if args.visual_check_interval and batch_idx % args.visual_check_interval == 0:
                grid = make_grid(reconstructed_images, nrow=args.n_iter + 1)
                save_image(grid, os.path.join(script_dir, args.output_dir, 'reconstructed', f'reconstructed_{accelerator.process_index}_{batch_idx}.png'))

    return ms_tuples

def main():
    args = create_argparser().parse_args()

    accelerator = Accelerator()
    accelerator.print(f"Using {accelerator.num_processes} GPUs.")

    loss_fn = lpips.LPIPS(net='vgg').to(accelerator.device)

    accelerator.print("Loading Stable Diffusion model...")
    model_id = args.model_id
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=th.float16 if args.use_fp16 else th.float32)
    
    pipe = pipe.to(accelerator.device)
    if not accelerator.is_main_process:
        pipe.set_progress_bar_config(disable=True)
        
    accelerator.print("Creating dataset...")
    dataset = OccupationDataset(args.data_dir, num_occupations=args.num_occupations)
    accelerator.print(f"Total of {len(dataset)} images") 
    
    if accelerator.is_main_process:
        os.makedirs(os.path.join(script_dir, args.output_dir), exist_ok=True)
        if args.visual_check_interval:
            os.makedirs(os.path.join(script_dir, args.output_dir, 'reconstructed'), exist_ok=True)

    accelerator.wait_for_everyone()
    accelerator.print("Computing minority scores and saving reconstructions...")
    with accelerator.split_between_processes(list(range(len(dataset)))) as dataset_idcs:
        local_dataset=dataset.select(dataset_idcs)
        
        print('-'*40)
        print(f"GPU{accelerator.process_index} working on {len(local_dataset)} entries")
        ms_tuples = compute_minority_scores(args, local_dataset, pipe, loss_fn, accelerator)

    ms_tuples = [tuple_ele for process in gather_object([ms_tuples]) for tuple_ele in process]
    print(f"MS gathered list length: {len(ms_tuples)}")
    
    if accelerator.is_main_process:
        # Speichere die Indizes, Prompts und Scores
        indices, prompts, scores = zip(*ms_tuples)
        scores = [score.cpu().numpy() for score in list(scores)]
        indices = list(indices)
        prompts = list(prompts)

        # Lade die Bilder basierend auf den gespeicherten Indizes
        images = [dataset[idx][0] for idx in indices]

        output_path = os.path.join(script_dir, args.output_dir)
        os.makedirs(output_path, exist_ok=True)

        # Speichere die Metadaten in einer CSV-Datei
        metadata = []

        for idx, image in enumerate(images):
            image_tensor = ToTensor()(image)
            image_filename = f'{idx}.png'
            save_image(image_tensor, os.path.join(output_path, image_filename))
            metadata.append([idx, prompts[idx], scores[idx]])

        # Speichere die Metadaten in einer CSV-Datei
        with open(os.path.join(output_path, 'metadata.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['idx', 'prompt', 'score'])
            writer.writerows(metadata)

    accelerator.wait_for_everyone()
    print("Dataset construction complete")

def create_argparser():
    defaults = dict(
        batch_size=25,
        use_fp16=True,
        data_dir="dataset_2",
        output_dir="dataset_2_ms",
        model_id="SG161222/Realistic_Vision_V2.0",
        ms_compute_only=False,
        n_iter=1,
        visual_check_interval=4,
        num_occupations=1,
    )
    parser = argparse.ArgumentParser()
    for k, v in defaults.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    return parser

if __name__ == "__main__":
    main()