import argparse
import os
import sys
import json
import random
import numpy as np
import torch as th
from tqdm import tqdm
import torch.distributed as dist
from diffusers import StableDiffusionPipeline

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, project_root)

from guided_diffusion import dist_util, logger

def load_occupations(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data['occupations']

def generate_prompts(occupation, batch_size):
    normal_prompts = [f"A photo of a {occupation}"] * batch_size
    sensitive_attributes = random.choices(["female", "male"], k=batch_size)
    sensitive_prompts = [f"A photo of a {attr} {occupation}" for attr in sensitive_attributes]
    return normal_prompts, sensitive_prompts

def main():
    args = create_argparser().parse_args()
    
    if not args.seed == "":
        th.manual_seed(int(args.seed))
        np.random.seed(int(args.seed))
    random.seed(int(args.seed)) if args.seed else None

    dist_util.setup_dist()
    
    # Lade Occupations
    occupations = load_occupations(os.path.join(script_dir, "occupations.json"))
    
    # Erstelle das Output-Verzeichnis relativ zum Skriptverzeichnis
    output_dir = os.path.join(script_dir, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Konfiguriere den Logger mit dem angegebenen Output-Verzeichnis
    logger.configure(dir=output_dir, format_strs=["log"])  # Nur die log.txt Datei

    logger.log("Loading Stable Diffusion model...")
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=th.float16 if args.use_fp16 else th.float32)
    pipe = pipe.to(dist_util.dev())

    # Verteile die Arbeit auf GPUs
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    local_occupations = occupations[rank::world_size]

    # Initialisiere den Fortschrittsbalken nur auf dem Hauptprozess
    if rank == 0:
        total_images = args.num_samples * len(occupations)
        pbar = tqdm(total=total_images, desc="Generating images", unit="img")

    # Speicherstruktur: ein Dict pro Occupation
    dataset = {occ: {"images": [], "sensitive_prompts": [], "normal_prompts": []} for occ in local_occupations}

    for occupation in local_occupations:
        for batch_start in range(0, args.num_samples, args.batch_size):
            batch_size = min(args.batch_size, args.num_samples - batch_start)
            normal_prompts, sensitive_prompts = generate_prompts(occupation, batch_size)
            
            with th.no_grad():
                images = pipe(sensitive_prompts, num_inference_steps=50, guidance_scale=7.5).images

            for img, normal_prompt, sensitive_prompt in zip(images, normal_prompts, sensitive_prompts):
                dataset[occupation]["images"].append(img)
                dataset[occupation]["sensitive_prompts"].append(sensitive_prompt)
                dataset[occupation]["normal_prompts"].append(normal_prompt)
            
            # Aktualisiere den Fortschrittsbalken nur auf dem Hauptprozess
            if rank == 0:
                pbar.update(batch_size * world_size)

    # Schließe den Fortschrittsbalken auf dem Hauptprozess
    if rank == 0:
        pbar.close()

    # Jeder Prozess speichert seinen Teil des Datensatzes
    logger.log(f"\nSaving dataset part to {output_dir}...")
    image_counter = 0
    for occupation, data in dataset.items():
        occ_dir = os.path.join(output_dir, occupation)
        os.makedirs(occ_dir, exist_ok=True)
        
        # Speichere Bilder
        for img in data["images"]:
            img.save(os.path.join(occ_dir, f"{image_counter}.png"))
            image_counter += 1
        
        # Speichere Prompts in einer CSV-Datei
        with open(os.path.join(occ_dir, f"prompts_{rank}.csv"), "w") as f:
            f.write("image_id,normal_prompt,sensitive_prompt\n")
            for i in range(len(data["images"])):
                f.write(f"{image_counter - len(data['images']) + i},{data['normal_prompts'][i]},{data['sensitive_prompts'][i]}\n")

    dist.barrier()
    logger.log("Dataset creation complete")

def create_argparser():
    defaults = dict(
        num_samples=50,  # Pro Occupation
        batch_size=4,
        use_fp16=True,
        seed="",
        output_dir="dataset"
    )
    parser = argparse.ArgumentParser()
    for k, v in defaults.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    return parser

if __name__ == "__main__":
    main()