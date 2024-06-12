import argparse
import os
import sys
import csv
import numpy as np
import torch as th
from torchvision.utils import save_image, make_grid
from torchvision.transforms import ToTensor, ToPILImage
from diffusers import StableDiffusionPipeline
from tqdm import tqdm
from PIL import Image
import lpips
from accelerate import Accelerator
from accelerate.utils import gather_object
from facenet_pytorch import MTCNN

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
        self.indices = []
        
        for idx, occupation in enumerate(self.occupations):
            occ_dir = os.path.join(self.data_dir, occupation)
            image_files = sorted(f for f in os.listdir(occ_dir) if f.endswith(".png"))
            self.image_files.extend(os.path.join(occ_dir, f) for f in image_files)
            
            prompts = read_csv_with_commas(os.path.join(occ_dir, "prompts.csv"), occupation)
            self.prompts.extend(prompts)
            self.indices.extend(range(len(self.image_files) - len(image_files), len(self.image_files)))
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        prompt, _ = self.prompts[idx]
        original_idx = self.indices[idx]
        image = Image.open(image_file).convert("RGB")
        return image, prompt, original_idx
    
    def select(self, indices):
        selected_dataset = OccupationDataset(self.data_dir)
        selected_dataset.image_files = [self.image_files[i] for i in indices]
        selected_dataset.prompts = [self.prompts[i] for i in indices]
        selected_dataset.indices = [self.indices[i] for i in indices]
        return selected_dataset

def collate_fn(batch):
    images = [item[0] for item in batch]
    prompts = [item[1] for item in batch]
    original_indices = [item[2] for item in batch]
    images = th.stack([ToTensor()(image) for image in images])
    return images, prompts, original_indices


def detect_faces(images, mtcnn):
    # Konvertiere die Bilder in PIL-Bilder
    pil_images = [ToPILImage()(img) for img in images]
    
    # Erkenne Gesichter in den Bildern und erhalte die Bounding Boxes
    faces = []
    boxes = []
    for image in pil_images:
        image_boxes, _ = mtcnn.detect(image)
        if image_boxes is not None:
            image_faces = mtcnn(image)
            faces.append(th.unsqueeze(image_faces[0],0))
            boxes.append(image_boxes[0])
        else:
          faces.append(None)
          boxes.append(None)
    
    return faces, boxes

def compute_minority_scores(args, global_dataset, local_dataset, pipe, loss_fn, accelerator):
    ms_tuples = []

    dataloader = th.utils.data.DataLoader(local_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    
    mtcnn = MTCNN(keep_all=True, device=accelerator.device)
    
    with th.no_grad():
        for batch_idx, (images, prompts, original_indices) in enumerate(tqdm(dataloader)):
            print(f"Process: {accelerator.process_index}, {prompts[0]}")
            
            # Konvertiere die Bilder in den richtigen Datentyp
            images = images.to(accelerator.device, dtype=pipe.vae.encoder.conv_in.bias.dtype)
            
            # Erkenne Gesichter in den Originalbildern
            original_faces, _ = detect_faces(images, mtcnn)
            
            # Filtere Bilder ohne erkannte Gesichter heraus
            face_indices = [i for i, face in enumerate(original_faces) if face is not None]
            
            images = images[face_indices]
            original_faces = [face for face in original_faces if face is not None]
            prompts = [prompts[i] for i in face_indices]
            original_indices = [original_indices[i] for i in face_indices]
            
            batch_losses = [[] for _ in range(len(images))]
            if args.visual_check_interval:
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
                
                # Erkenne Gesichter in den rekonstruierten Bildern
                denoised_faces, _ = detect_faces(denoised_images, mtcnn)
                
                # Berechne den LPIPS-Loss für jedes Bild mit erkanntem Gesicht
                for j, (original_face, denoised_face) in enumerate(zip(original_faces, denoised_faces)):
                    if denoised_face is not None:
                        #grid = make_grid(th.cat((original_face, denoised_face)), 1)
                        #save_image(grid, os.path.join(script_dir, args.output_dir, 'testing', f'test_{i}_{j}.png'))
                        LPIPS_loss = loss_fn(original_face.to(accelerator.device), denoised_face.to(accelerator.device))
                        batch_losses[j].append(LPIPS_loss.item())
                
                if args.visual_check_interval:
                    reconstructed_images[len(images) * (i + 1):len(images) * (i + 2)] = denoised_images

            # Filtere Bilder ohne Losses heraus und berechne den Mittelwert der Losses für jedes Bild
            filtered_indices = [i for i, losses in enumerate(batch_losses) if len(losses) > 0]
            filtered_prompts = [prompts[i] for i in filtered_indices]
            filtered_original_indices = [original_indices[i] for i in filtered_indices]
            filtered_batch_losses = [np.mean(losses) for losses in batch_losses if len(losses) > 0]

            # Speichere die ursprünglichen Indizes zusammen mit Prompts und Scores
            for idx, (prompt, score, original_idx) in enumerate(zip(filtered_prompts, filtered_batch_losses, filtered_original_indices)):
                ms_tuples.append((original_idx, prompt, score))

            if args.visual_check_interval and (batch_idx + 1) % args.visual_check_interval == 0:
                grid = make_grid(reconstructed_images, nrow=args.n_iter + 1)
                save_image(grid, os.path.join(script_dir, args.output_dir, 'reconstructed', f'reconstructed_{accelerator.process_index}_{batch_idx}.png'))

            # Überprüfe, ob alle GPUs x Occupations fertig erstellt haben
            if (batch_idx + 1) % args.save_interval == 0:
                # Speichere die Daten periodisch
                save_data(args, global_dataset, ms_tuples, accelerator)
                ms_tuples = []  # Leere die Liste, um Speicher freizugeben
    
    return ms_tuples
  

def save_data(args, dataset, ms_tuples, accelerator):
    gathered_ms_tuples = [tuple_ele for process in gather_object([ms_tuples]) for tuple_ele in process]
    print(f"MS gathered list length: {len(gathered_ms_tuples)}")
    
    if accelerator.is_main_process:
        # Speichere die Indizes, Prompts und Scores
        indices = []
        prompts = []
        scores = []
        for index, prompt, score in gathered_ms_tuples:
            indices.append(index)
            prompts.append(prompt)
            scores.append(score)

        # Lade die Bilder basierend auf den gespeicherten Indizes
        images = [dataset[idx][0] for idx in indices]

        output_path = os.path.join(script_dir, args.output_dir)
        os.makedirs(output_path, exist_ok=True)

        # Speichere die Metadaten in einer CSV-Datei
        metadata = []

        for idx, (image, prompt, score) in enumerate(zip(images, prompts, scores)):
            image_tensor = ToTensor()(image)
            image_filename = f'{indices[idx]}.png'  # Verwende den ursprünglichen Index als Dateinamen
            save_image(image_tensor, os.path.join(output_path, image_filename))
            metadata.append([indices[idx], prompt, score])

        # Speichere die Metadaten in einer CSV-Datei
        with open(os.path.join(output_path, 'metadata.csv'), 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(metadata)

    accelerator.wait_for_everyone()

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
        ms_tuples = compute_minority_scores(args, dataset, local_dataset, pipe, loss_fn, accelerator)

    # Speichere die verbleibenden Daten
    save_data(args, dataset, ms_tuples, accelerator)
    
    accelerator.wait_for_everyone()
    print("Dataset construction complete")

def create_argparser():
    defaults = dict(
        batch_size=20,
        use_fp16=True,
        data_dir="dataset_2",
        output_dir="dataset_2_ms_faces",
        model_id="SG161222/Realistic_Vision_V2.0",
        ms_compute_only=False,
        n_iter=5,
        visual_check_interval=None,
        num_occupations=None,
        save_interval=1,  # Füge das Argument für das Speicherintervall hinzu
    )
    parser = argparse.ArgumentParser()
    for k, v in defaults.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    return parser

if __name__ == "__main__":
    main()