import csv
import argparse
import os
import sys
import json
import random
import numpy as np
import torch as th
from tqdm import tqdm
from accelerate import Accelerator
from torchvision.transforms import ToTensor
from facenet_pytorch import MTCNN
from utils.custom_pipe import HDiffusionPipeline
from accelerate.utils import gather_object

from utils.semdiff import StableSemanticDiffusion, ConditionalUnet
from diffusers import StableDiffusionPipeline, DDIMScheduler

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, project_root)

from utils.face_detection import detect_faces
from utils.fairface import load_fairface_model, predict_gender

TIMESTEPS = [981, 961, 941, 921, 901, 881, 861, 841, 821, 801, 781, 761, 741,
             721, 701, 681, 661, 641, 621, 601, 581, 561, 541, 521, 501, 481, 461,
             441, 421, 401, 381, 361, 341, 321, 301, 281, 261, 241, 221, 201, 181,
             161, 141, 121, 101, 81, 61, 41, 21, 1]

def load_occupations(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data['occupations']

def generate_all_prompts(occupations, num_samples):
    all_prompts = []
    for occupation in occupations:
        attributes = random.choices(["female", "male"], k=num_samples)
        prompts = [f"A photo of the face of a {attr} {occupation}" for attr in attributes]
        all_prompts.extend(prompts)
    return all_prompts

def get_occupation_from_prompt(prompt):
    words = prompt.split()
    return words[-1]  # Das letzte Wort ist die Occupation

def process_batch(args, prompts, pipe, fairface, accelerator, mtcnn):
    batch_data = []

    with th.no_grad():
        outputs = pipe.sample(prompts=prompts, enable_prog_bar=not accelerator.is_main_process)
        #images, h_vects = pipe(prompts, num_inference_steps=50, guidance_scale=7.5, return_dict=False)
        images = outputs.x0
        h_vects = outputs.hs

    for idx in range(len(images)):
        faces, _ = detect_faces([images[idx]], mtcnn)
        face = faces[0]

        if face is not None:
            gender_scores = predict_gender(face, fairface, accelerator.device)
        else:
            gender_scores = []

        batch_data.append({
            'image': images[idx],
            'prompt': prompts[idx],
            'occupation': get_occupation_from_prompt(prompts[idx]),
            'h_vects': {int(ts): h_vects[idx][int(ts)].cpu().numpy() for ts in pipe.diff.timesteps},
            'face_detected': face is not None,
            'gender_scores': gender_scores
        })

    return batch_data

def save_data(output_path, data, accelerator, image_counter):
    print(f"SAVING DATA, on process: {accelerator.process_index}, num_images: {len(data)}")
    h_vects_dir = os.path.join(output_path, 'h_vects')
    os.makedirs(h_vects_dir, exist_ok=True)

    rows_to_write = []
    for idx, item in enumerate(data):
        # Erstelle einen eindeutigen Dateinamen für jedes Bild
        image_filename = f"image_{accelerator.process_index}_{image_counter}.png"
        image_dir = os.path.join(output_path, "images")
        os.makedirs(image_dir, exist_ok=True)
        image_path = os.path.join(image_dir, image_filename)

        # Speichere das Bild
        item['image'].save(image_path)

        # Speichere h_vects
        h_vects_filename = f"h_vects_{accelerator.process_index}_{image_counter}.npz"
        h_vects_path = os.path.join(h_vects_dir, h_vects_filename)
        
        # Konvertiere die Zeitschritte in Strings
        h_vects_dict = {str(ts): arr for ts, arr in item['h_vects'].items()}
        np.savez_compressed(h_vects_path, **h_vects_dict)

        # Bereite die Zeile für die CSV-Datei vor
        rows_to_write.append([
            image_filename,
            item['prompt'],
            item['occupation'],
            h_vects_filename,
            item['face_detected'],
            ','.join(map(str, item['gender_scores'])) if item['gender_scores'] else ''
        ])

        image_counter += 1

    return rows_to_write, image_counter

def main():
    args = create_argparser().parse_args()

    accelerator = Accelerator()
    accelerator.print(f"Using {accelerator.num_processes} GPUs.")

    fairface = load_fairface_model(accelerator.device)

    accelerator.print("Loading Diffusion model...")
    model_id = args.model_id
    pipe = StableDiffusionPipeline.from_pretrained(model_id).to(accelerator.device) 
    scheduler = pipe.scheduler

    pipe = StableSemanticDiffusion(
        unet=ConditionalUnet(pipe.unet),
        scheduler=DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False),
        vae = pipe.vae,
        tokenizer = pipe.tokenizer,
        text_encoder = pipe.text_encoder,
        image_processor = pipe.image_processor,
        model_id = model_id,
        num_inference_steps=50
    )
    
    #pipe = HDiffusionPipeline.from_pretrained(model_id, torch_dtype=th.float16 if args.use_fp16 else th.float32)
    #pipe = pipe.to(accelerator.device)
    
    #if not accelerator.is_main_process:
    #    pipe.diff.set_progress_bar_config(disable=True)

    accelerator.print("Creating dataset...")
    occupations = load_occupations(os.path.join(script_dir, args.occupations_file))
    all_prompts = generate_all_prompts(occupations, args.num_samples)

    output_path = os.path.join(script_dir, args.output_dir)
    
    # Erstelle das Ausgabeverzeichnis für alle Prozesse
    os.makedirs(output_path, exist_ok=True)
    
    # Warte, bis alle Prozesse das Verzeichnis erstellt haben
    accelerator.wait_for_everyone()

    mtcnn = MTCNN(keep_all=True, device=accelerator.device)

    with accelerator.split_between_processes(all_prompts) as local_prompts:
        data = []
        all_rows = []
        image_counter = 0

        for i in tqdm(range(0, len(local_prompts), args.batch_size), desc="Processing batches", disable=not accelerator.is_main_process):
            batch_prompts = local_prompts[i:i+args.batch_size]
            batch_data = process_batch(args, batch_prompts, pipe, fairface, accelerator, mtcnn)
            data.extend(batch_data)

            # Save data periodically to free up memory
            if len(data) >= args.save_interval:
                rows, image_counter = save_data(output_path, data, accelerator, image_counter)
                all_rows.extend(rows)
                data = []

        # Save remaining data
        if len(data) > 0:
            rows, _ = save_data(output_path, data, accelerator, image_counter)
            all_rows.extend(rows)

    # Gather all rows from all processes
    accelerator.wait_for_everyone()
    all_gathered_rows = gather_object(all_rows)

    # Write to CSV file (only on main process)
    if accelerator.is_main_process:
        csv_filename = "metadata.csv"
        csv_path = os.path.join(output_path, csv_filename)
        with open(csv_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['image_filename', 'prompt', 'occupation', 'h_vects_filename', 'face_detected', 'gender_scores'])
            for row in all_gathered_rows:
                row[4] = int(row[4])  # Turn detected_face boolean to a 0/1 int
                csv_writer.writerow(row)

    print(f"Dataset construction complete, process {accelerator.process_index}")

def create_argparser():
    defaults = dict(
        num_samples=200,
        batch_size=8,
        use_fp16=True,
        occupations_file="occupations.json",
        output_dir="output",
        model_id="SG161222/Realistic_Vision_V2.0", #"runwayml/stable-diffusion-v1-5",
        save_interval=16,
    )
    parser = argparse.ArgumentParser()
    for k, v in defaults.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    return parser

if __name__ == "__main__":
    main()