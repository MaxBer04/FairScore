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

def compute_minority_scores(args, local_dict, pipe, loss_fn, accelerator):
    ms_dict = dict(images=[], scores=[])
    
    for idx, (images, prompts) in enumerate(zip(local_dict["images"], local_dict["prompts"])):
        #print(f"Process {accelerator.process_index}: {prompts[0]}")
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

            LPIPS_loss = loss_fn(images, denoised_images)
            batch_losses[:, i] = LPIPS_loss.view(-1)
            reconstructed_images[len(images) * (i + 1):len(images) * (i + 2)] = denoised_images

        ms_dict["scores"].extend(batch_losses.mean(dim=1))
        ms_dict["images"].extend(images)

        if accelerator.process_index == 3 and (i+1) % args.visual_check_interval == 0:
            grid = make_grid(reconstructed_images, nrow=args.n_iter + 1)
            save_image(grid, os.path.join(script_dir, args.output_dir, f'reconstructed_{accelerator.process_index}_{idx}.png'))

    #ms_list = th.cat(ms_list)
    return ms_dict

def main():
    args = create_argparser().parse_args()

    accelerator = Accelerator()
    print(f"Using {accelerator.num_processes} GPUs.")

    loss_fn = lpips.LPIPS(net='vgg').to(accelerator.device)

    print("Loading Stable Diffusion model...")
    model_id = args.model_id
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=th.float16 if args.use_fp16 else th.float32)
    
    pipe = pipe.to(accelerator.device)
    if not accelerator.is_main_process:
        pipe.set_progress_bar_config(disable=True)
        
    print("Creating dataset...")
    dataset = OccupationDataset(args.data_dir, num_occupations=1)
    
    if accelerator.is_main_process:
        os.makedirs(os.path.join(script_dir, args.output_dir), exist_ok=True)
        print("Computing minority scores and saving reconstructions...")

    dataloader = th.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    iter_dict = {"images": [], "prompts": []}
    for images, prompts in dataloader:
        iter_dict["images"].append(images)
        iter_dict["prompts"].append(prompts)
        
    accelerator.wait_for_everyone()
    with accelerator.split_between_processes(iter_dict) as local_dict:
        ms_dict = compute_minority_scores(args, local_dict, pipe, loss_fn, accelerator)

    #accelerator.wait_for_everyone()
    ms = gather_object([ms_dict])
    ms = th.cat(ms).cpu().float()

    if accelerator.is_main_process:
        print(len(ms))
        th.save(ms, os.path.join(script_dir, args.output_dir, 'ms_values.pt'))

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

        output_path_train = os.path.join(script_dir, args.output_dir, "train")
        output_path_val = os.path.join(script_dir, args.output_dir, "val")
        os.makedirs(output_path_train, exist_ok=True)
        os.makedirs(output_path_val, exist_ok=True)

        for index, image in enumerate(tqdm(iter_dict["images"])):

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
        val_ratio=0.05,
        n_iter=5,
        num_m_classes=2,
        visual_check_interval=1
    )
    parser = argparse.ArgumentParser()
    for k, v in defaults.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    return parser

if __name__ == "__main__":
    main()