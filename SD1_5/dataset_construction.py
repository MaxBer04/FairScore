import argparse
import os
import sys
import json
import random
import numpy as np
import torch as th
from tqdm import tqdm
from diffusers import StableDiffusionPipeline
from accelerate import Accelerator

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, project_root)

from guided_diffusion import logger

def load_occupations(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data['occupations']

def generate_prompts(occupation, batch_size):
    normal_prompts = [f"A photo of a {occupation}"] * batch_size
    sensitive_attributes = random.choices(["female", "male"], k=batch_size)
    sensitive_prompts = [f"A photo of a {attr} {occupation}" for attr in sensitive_attributes]
    return normal_prompts, sensitive_prompts

def save_dataset(dataset, output_dir, image_counter):
    logger.log(f"\nSaving dataset to {output_dir}...")
    for occupation, data in dataset.items():
        occ_dir = os.path.join(output_dir, occupation)
        os.makedirs(occ_dir, exist_ok=True)
        
        # Speichere Bilder
        for img in data["images"]:
            img.save(os.path.join(occ_dir, f"{image_counter}.png"))
            image_counter += 1
        
        # Speichere Prompts in einer CSV-Datei
        with open(os.path.join(occ_dir, "prompts.csv"), "w") as f:
            f.write("image_id,normal_prompt,sensitive_prompt\n")
            for i in range(len(data["images"])):
                f.write(f"{image_counter - len(data['images']) + i},{data['normal_prompts'][i]},{data['sensitive_prompts'][i]}\n")
    
    return image_counter

def main():
    args = create_argparser().parse_args()
    
    if not args.seed == "":
        th.manual_seed(int(args.seed))
        np.random.seed(int(args.seed))
    random.seed(int(args.seed)) if args.seed else None

    accelerator = Accelerator()
    print('-'*32)
    print(f"Using {accelerator.num_processes} GPUs.")
    print('-'*32)
    
    # Lade Occupations
    occupations = load_occupations(os.path.join(script_dir, "occupations.json"))
    
    # Erstelle das Output-Verzeichnis relativ zum Skriptverzeichnis
    output_dir = os.path.join(script_dir, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Konfiguriere den Logger mit dem angegebenen Output-Verzeichnis
    logger.configure(dir=output_dir, format_strs=["log"])  # Nur die log.txt Datei

    logger.log("Loading Stable Diffusion model...")
    model_id = args.model_id
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=th.float16 if args.use_fp16 else th.float32)
    
    # Move the model to device before preparing
    pipe = pipe.to(accelerator.device)
    pipe.set_progress_bar_config(disable=True)

    # Verteile das Modell auf die verfügbaren GPUs
    pipe = accelerator.prepare(pipe)

    # Initialisiere den Fortschrittsbalken
    total_images = args.num_samples * len(occupations)
    progress_bar = tqdm(total=total_images, desc="Generating images", disable=not accelerator.is_local_main_process)

    # Speicherstruktur: ein Dict pro Occupation
    dataset = {occ: {"images": [], "sensitive_prompts": [], "normal_prompts": []} for occ in occupations}

    # Verteile die Occupations auf die verfügbaren GPUs
    occupations_per_gpu = len(occupations) // accelerator.num_processes
    occupations_remainder = len(occupations) % accelerator.num_processes
    occupations_split = [occupations_per_gpu] * accelerator.num_processes
    for i in range(occupations_remainder):
        occupations_split[i] += 1
    occupations_split = np.cumsum(occupations_split)
    start_idx = 0 if accelerator.process_index == 0 else occupations_split[accelerator.process_index - 1]
    end_idx = occupations_split[accelerator.process_index]
    local_occupations = occupations[start_idx:end_idx]
    print(f"Process {accelerator.process_index} handling occupations: {local_occupations}")

    image_counter = 0
    save_interval = 2
    for idx, occupation in enumerate(local_occupations, start=1):
        for batch_start in range(0, args.num_samples, args.batch_size):
            batch_size = min(args.batch_size, args.num_samples - batch_start)
            normal_prompts, sensitive_prompts = generate_prompts(occupation, batch_size)
            
            # Move prompts to device
            sensitive_prompts = [prompt.replace("<", "").replace(">", "") for prompt in sensitive_prompts]
            
            with th.no_grad():
                images = pipe(sensitive_prompts, num_inference_steps=50, guidance_scale=7.5).images

            for img, normal_prompt, sensitive_prompt in zip(images, normal_prompts, sensitive_prompts):
                dataset[occupation]["images"].append(img)
                dataset[occupation]["sensitive_prompts"].append(sensitive_prompt)
                dataset[occupation]["normal_prompts"].append(normal_prompt)
            
            progress_bar.update(batch_size * accelerator.num_processes)
        
        # Speichere Bilder nach der Erstellung von 10 Occupations
        if idx % save_interval == 0:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                image_counter = save_dataset(dataset, output_dir, image_counter)
            dataset = {occ: {"images": [], "sensitive_prompts": [], "normal_prompts": []} for occ in occupations}
            accelerator.wait_for_everyone()

    # Schließe den Fortschrittsbalken
    progress_bar.close()

    # Speichere den restlichen Datensatz
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_dataset(dataset, output_dir, image_counter)

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.log("Dataset creation complete")

def create_argparser():
    defaults = dict(
        num_samples=100,  # Pro Occupation
        batch_size=32,
        use_fp16=True,
        seed="",
        output_dir="dataset_2",
        model_id="SG161222/Realistic_Vision_V2.0"  # "runwayml/stable-diffusion-v1-5"
    )
    parser = argparse.ArgumentParser()
    for k, v in defaults.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    return parser

if __name__ == "__main__":
    main()