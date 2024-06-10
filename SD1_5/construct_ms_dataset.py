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
from mpi4py import MPI
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, project_root)

from guided_diffusion import dist_util

class OccupationDataset(th.utils.data.Dataset):
    def __init__(self, data_dir):
        self.data_dir = os.path.join(script_dir, data_dir)
        self.occupations = [d for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d))]
        self.image_files = []
        self.prompts = []
        
        for occupation in self.occupations:
            occ_dir = os.path.join(self.data_dir, occupation)
            image_files = sorted(f for f in os.listdir(occ_dir) if f.endswith(".png"))
            self.image_files.extend(os.path.join(occ_dir, f) for f in image_files)
            
            with open(os.path.join(occ_dir, "prompts_0.csv"), "r") as f:
                f.readline()  # Skip header
                for line in f:
                    _, normal_prompt, _ = line.strip().split(",")
                    self.prompts.append(normal_prompt)
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        prompt = self.prompts[idx]
        image = Image.open(image_file).convert("RGB")
        return image, prompt


def compute_minority_scores(args, dataset, pipe, loss_fn):
    ms_list = []
    to_tensor = ToTensor()

    with th.no_grad():
        for index, (image, prompt) in enumerate(tqdm(dataset)):
            if MPI.COMM_WORLD.Get_rank() != index % MPI.COMM_WORLD.Get_size():
                continue

            image = to_tensor(image).unsqueeze(0).to(dist_util.dev())
            
            # Tokenize the prompt and convert it to a tensor
            #prompt_input = pipe.tokenizer(prompt, padding="max_length", max_length=pipe.tokenizer.model_max_length, truncation=True, return_tensors="pt")
            #prompt_input = {k: v.to(dist_util.dev()) for k, v in prompt_input.items()}

            image_losses = th.zeros(args.n_iter)
            reconstructed_images = []
            
            latents = pipe.vae.encode(image).latent_dist.sample().detach()
            latents = latents * pipe.vae.config.scaling_factor

            for i in range(args.n_iter):
                timestep = 600#int(0.9 * pipe.scheduler.config.num_train_timesteps)
                timesteps = th.tensor([timestep], dtype=th.long).to(dist_util.dev())
                noise = th.randn_like(latents)
                noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)
                
                # Pass the tokenized prompt to the text encoder
                denoised_image = pipe(prompt, latents=noisy_latents).images[0]
                denoised_image = to_tensor(denoised_image).unsqueeze(0).to(dist_util.dev())
                #model_output = pipe.unet(noisy_latents, timesteps, encoder_hidden_states=pipe.text_encoder(**prompt_input)[0]).sample
                
                #denoised_image = pipe.vae.decode(model_output).sample

                LPIPS_loss = loss_fn(image, denoised_image)
                image_losses[i] = LPIPS_loss.item()
                reconstructed_images.append(denoised_image)

            ms_list.append(image_losses.mean())
            print(ms_list)
            # Save the original image and reconstructed images side by side
            reconstructed_images = th.cat([image] + reconstructed_images, dim=0)
            grid = make_grid(reconstructed_images, nrow=args.n_iter+1)
            #grid = (grid * 0.5 + 0.5).clamp_(0.0, 1.0)
            save_image(grid, os.path.join(script_dir, args.output_dir, f'reconstructed_{index}.png'))

    return ms_list

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    
    loss_fn = lpips.LPIPS(net='vgg').to(dist_util.dev())

    print("Loading Stable Diffusion model...")
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id)
    pipe = pipe.to(dist_util.dev())
    if args.use_fp16:
        pipe = pipe.to(torch_dtype=th.float16)

    print("Creating dataset...")
    dataset = OccupationDataset(args.data_dir)
    
    if MPI.COMM_WORLD.Get_rank() == 0:
        os.makedirs(os.path.join(script_dir, args.output_dir), exist_ok=True)

    print("Computing minority scores and saving reconstructions...")
    ms_list = compute_minority_scores(args, dataset, pipe, loss_fn)

    ms = th.tensor(ms_list).cpu()

    if MPI.COMM_WORLD.Get_rank() == 0:
        th.save(ms, os.path.join(script_dir, args.output_dir, 'ms_values.pt'))

    if args.ms_compute_only:
    #    dist_util.barrier()
        print("Minority score computation and image reconstruction complete")
        sys.exit()

    print("Constructing dataset labeled with minority scores...")

    q_th = th.zeros(args.num_m_classes)
    for i in range(len(q_th)):
        q_th[i] = th.quantile(ms, 1 / args.num_m_classes * (i+1))
    
    ms_labels = th.zeros_like(ms).long()

    for i in range(len(ms)):
        current = ms[i]
        for j in range(len(q_th)):
            if j == 0:
                if current <= q_th[j]:
                    ms_labels[i] = j
            else:
                if current > q_th[j-1] and current <= q_th[j]:
                    ms_labels[i] = j

    data_indices = th.arange(len(ms_labels))
    train_indices, val_indices, y_train, y_val = train_test_split(data_indices, ms_labels, test_size=args.val_ratio, stratify=ms_labels)

    if dist_util.get_rank() == 0:
        output_path_train = os.path.join(script_dir, args.output_dir, "train")
        output_path_val = os.path.join(script_dir, args.output_dir, "val")
        os.makedirs(output_path_train, exist_ok=True)
        os.makedirs(output_path_val, exist_ok=True)

    for index, (image, _) in enumerate(tqdm(dataset)):
        if dist_util.get_rank() != index % dist_util.get_world_size():
            continue

        if index in train_indices:
            label_index = np.where(train_indices==index)[0].item()
            label = y_train[label_index]
            image = (image * 0.5 + 0.5).clamp_(0.0, 1.0)
            save_image(image, os.path.join(output_path_train, f'{label:04d}_{index}.png'))
        else:
            label_index = np.where(val_indices==index)[0].item()
            label = y_val[label_index]
            image = (image * 0.5 + 0.5).clamp_(0.0, 1.0)
            save_image(image, os.path.join(output_path_val, f'{label:04d}_{index}.png'))

    dist_util.barrier()
    print("Dataset construction complete")

def create_argparser():
    defaults = dict(
        use_fp16=True,
        data_dir="dataset",
        output_dir="out-2",
        ms_compute_only=True,
        val_ratio=0.05,
        n_iter=5,
        num_m_classes=100,
    )
    parser = argparse.ArgumentParser()
    for k, v in defaults.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    return parser

if __name__ == "__main__":
    main()