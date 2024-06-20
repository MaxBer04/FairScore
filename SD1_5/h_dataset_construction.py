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

def generate_prompts(occupation, batch_size):
    attributes = random.choices(["female", "male"], k=batch_size)
    prompts = [f"A photo of the face of a {attr} {occupation}" for attr in attributes]
    return prompts, attributes

def process_batch(args, prompts, pipe, fairface, accelerator, mtcnn):
    batch_data = []

    with th.no_grad():
        images, h_vects = pipe(prompts, num_inference_steps=50, guidance_scale=7.5, return_dict=False)

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
            'h_vects': {ts: h_vects[ts][2*idx:2*idx+2].cpu().numpy() for ts in TIMESTEPS},
            'face_detected': face is not None,
            'gender_scores': gender_scores
        })

    return batch_data

def save_data(output_path, data, accelerator):
    print(f"SAVING DATA, on process: {accelerator.process_index}, num_images: {len(data)}")
    h_vects_dir = os.path.join(output_path, 'h_vects')
    os.makedirs(h_vects_dir, exist_ok=True)

    for idx, item in enumerate(data):
        # Erstelle einen eindeutigen Dateinamen für jedes Bild
        image_filename = f"image_{accelerator.process_index}_{idx}.png"
        image_path = os.path.join(output_path, image_filename)

        # Speichere das Bild
        item['image'].save(image_path)

        # Speichere h_vects
        h_vects_filename = f"h_vects_{accelerator.process_index}_{idx}.npz"
        h_vects_path = os.path.join(h_vects_dir, h_vects_filename)
        
        # Konvertiere die Zeitschritte in Strings
        h_vects_dict = {str(ts): arr for ts, arr in item['h_vects'].items()}
        np.savez_compressed(h_vects_path, **h_vects_dict)

        # Erstelle den Dateinamen für die JSON-Datei
        json_filename = f"data_{accelerator.process_index}_{idx}.json"
        json_path = os.path.join(output_path, json_filename)

        # Erstelle das Daten-Dictionary für die JSON-Datei
        json_data = {
            'image_filename': image_filename,
            'prompt': item['prompt'],
            'h_vects_filename': h_vects_filename,
            'face_detected': item['face_detected'],
            'gender_scores': item['gender_scores'] if len(item['gender_scores']) > 0 else []
        }

        # Speichere die JSON-Datei
        with open(json_path, 'w') as f:
            json.dump(json_data, f)


def main():
    args = create_argparser().parse_args()

    accelerator = Accelerator()
    accelerator.print(f"Using {accelerator.num_processes} GPUs.")

    fairface = load_fairface_model(accelerator.device)

    accelerator.print("Loading H-Diffusion model...")
    model_id = args.model_id
    pipe = HDiffusionPipeline.from_pretrained(model_id, torch_dtype=th.float16 if args.use_fp16 else th.float32)

    pipe = pipe.to(accelerator.device)
    if not accelerator.is_main_process:
        pipe.set_progress_bar_config(disable=True)

    accelerator.print("Creating dataset...")
    occupations = load_occupations(os.path.join(script_dir, args.occupations_file))

    output_path = os.path.join(script_dir, args.output_dir)
    if accelerator.is_main_process:
        os.makedirs(output_path, exist_ok=True)

    mtcnn = MTCNN(keep_all=True, device=accelerator.device)

    with accelerator.split_between_processes(occupations) as local_occupations:
        data = []

        for occupation in tqdm(local_occupations, desc="Processing occupations", disable=not accelerator.is_main_process):
            for batch_start in range(0, args.num_samples, args.batch_size):
                batch_size = min(args.batch_size, args.num_samples - batch_start)
                prompts, attributes = generate_prompts(occupation, batch_size)
                batch_data = process_batch(args, prompts, pipe, fairface, accelerator, mtcnn)
                data.extend(batch_data)

                # Save data periodically to free up memory
                if len(data) >= args.save_interval:
                    save_data(output_path, data, accelerator)
                    data = []

        # Save remaining data
        if len(data) > 0:
            save_data(output_path, data, accelerator)

    print(f"Dataset construction complete, process {accelerator.process_index}")

def create_argparser():
    defaults = dict(
        num_samples=40,
        batch_size=40,
        use_fp16=True,
        occupations_file="occupations.json",
        output_dir="output",
        model_id="SG161222/Realistic_Vision_V2.0",
        save_interval=100,
    )
    parser = argparse.ArgumentParser()
    for k, v in defaults.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    return parser

if __name__ == "__main__":
    main()