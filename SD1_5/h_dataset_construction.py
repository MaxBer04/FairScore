import argparse
import os
import sys
import json
import random
import numpy as np
import torch as th
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import gather_object
from torchvision.transforms import functional as F
from torchvision.transforms import ToTensor
from facenet_pytorch import MTCNN
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, project_root)

from guided_diffusion import logger
from utils.face_detection import detect_faces
from utils.fairface import load_fairface_model, predict_gender
from utils.custom_pipe import HDiffusionPipeline


def load_occupations(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data['occupations']

def generate_prompts(occupation, batch_size):
    normal_prompts = [f"A photo of the face of a {occupation}"] * batch_size
    sensitive_attributes = random.choices(["female", "male"], k=batch_size)
    sensitive_prompts = [f"A photo of the face of a {attr} {occupation}" for attr in sensitive_attributes]
    return normal_prompts, sensitive_prompts

def save_dataset(dataset, output_dir):
    logger.log(f"\nSaving dataset to {output_dir}...")
    for occupation, data in dataset.items():
        if len(data["h_vects"]) > 0:
            occ_dir = os.path.join(output_dir, occupation)
            os.makedirs(occ_dir, exist_ok=True)
            
            # Speichere h_vects als NumPy-Arrays
            for i, h_vects in enumerate(data["h_vects"]):
                np.savez(os.path.join(occ_dir, f"{i}_h_vects.npz"), **{str(k): v for k, v in h_vects.items()})
            
            # Speichere Metadaten in einer JSON-Datei
            with open(os.path.join(occ_dir, "metadata.json"), "w") as f:
                json.dump({
                    "normal_prompts": data["normal_prompts"],
                    "sensitive_prompts": data["sensitive_prompts"],
                    "face_detected": data["face_detected"],
                    "gender_probs": data["gender_probs"]
                }, f)

def plot_images(images, faces, boxes, gender_probs, batch_idx, output_dir):
    num_images = len(images)
    if num_images > 10:
        idxs = random.sample(range(num_images), 10)
        images = [images[i] for i in idxs]
        faces = [faces[i] for i in idxs]
        boxes = [boxes[i] for i in idxs]
        gender_probs = [gender_probs[i] for i in idxs]

    fig, axs = plt.subplots(2, 5, figsize=(20, 8))
    axs = axs.ravel()

    for i, (img, face, box, gender_prob) in enumerate(zip(images, faces, boxes, gender_probs)):
        if not isinstance(img, th.Tensor):
          img = ToTensor()(img)
        img = img.cpu().numpy().transpose(1, 2, 0)
        img = (img * 255).astype(np.uint8)
        img = Image.fromarray(img)

        if face is not None:
            draw = ImageDraw.Draw(img)
            draw.rectangle(box.tolist(), outline="red", width=2)

        axs[i].imshow(img)
        if gender_prob:
            axs[i].set_title(f"Male: {gender_prob[0]:.2f}, Female: {gender_prob[1]:.2f}")
        else:
            axs[i].set_title("No face detected")
        axs[i].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"batch_{batch_idx}.png"))
    plt.close()

def main():
    args = create_argparser().parse_args()
    
    if not args.seed == "":
        th.manual_seed(int(args.seed))
        np.random.seed(int(args.seed))
    random.seed(int(args.seed)) if args.seed else None

    accelerator = Accelerator()
    if accelerator.is_main_process:
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
    pipe = HDiffusionPipeline.from_pretrained(model_id, torch_dtype=th.float16 if args.use_fp16 else th.float32)
    
    # Move the model to device before preparing
    pipe = pipe.to(accelerator.device)
    pipe.set_progress_bar_config(disable=True)
    
    # Verteile das Modell auf die verfügbaren GPUs
    pipe = accelerator.prepare(pipe)
    
    # Lade das MTCNN-Modell für die Gesichtserkennung
    mtcnn = MTCNN(keep_all=True, device=accelerator.device)
    
    # Lade das FairFace-Modell für die Geschlechtsklassifizierung
    fairface = load_fairface_model(accelerator.device)
    
    # Initialisiere den Fortschrittsbalken
    total_images = args.num_samples * len(occupations)
    progress_bar = tqdm(total=total_images, desc="Generating images", disable=not accelerator.is_local_main_process)

    # Verteile die Occupations auf die verfügbaren GPUs
    with accelerator.split_between_processes(occupations) as local_occupations:

        # Speicherstruktur: ein Dict pro Occupation
        dataset = {occ: {"h_vects": [], "sensitive_prompts": [], "normal_prompts": [], "face_detected": [], "gender_probs": []} for occ in local_occupations}

        for idx, occupation in enumerate(local_occupations, start=1):
            for batch_start in range(0, args.num_samples, args.batch_size):
                batch_size = min(args.batch_size, args.num_samples - batch_start)
                normal_prompts, sensitive_prompts = generate_prompts(occupation, batch_size)
                
                # Move prompts to device
                sensitive_prompts = [prompt.replace("<", "").replace(">", "") for prompt in sensitive_prompts]
                
                with th.no_grad():
                    images, h_vects = pipe(sensitive_prompts, num_inference_steps=50, guidance_scale=7.5, return_dict=False)

                # Erkenne Gesichter in den generierten Bildern
                faces, boxes = detect_faces(images, mtcnn)

                gender_probs_batch = []
                for face in faces:
                    if face is not None:
                        gender_probs = predict_gender(face, fairface, accelerator.device)
                        gender_probs_batch.append(gender_probs)
                    else:
                        gender_probs_batch.append([])

                for h_vect, normal_prompt, sensitive_prompt, face_detected, gender_prob in zip(h_vects, normal_prompts, sensitive_prompts, [face is not None for face in faces], gender_probs_batch):
                    dataset[occupation]["h_vects"].append(h_vect)
                    dataset[occupation]["sensitive_prompts"].append(sensitive_prompt)
                    dataset[occupation]["normal_prompts"].append(normal_prompt)
                    dataset[occupation]["face_detected"].append(face_detected)
                    dataset[occupation]["gender_probs"].append(gender_prob)
                
                progress_bar.update(batch_size*accelerator.num_processes)

                # Plotte Bilder nach jedem x-ten Batch
                if (batch_start // args.batch_size) % args.plot_every == 0:
                    plot_images(images, faces, boxes, gender_probs_batch, batch_start // args.batch_size, output_dir)

    progress_bar.close()

    accelerator.wait_for_everyone()
    gathered_dataset = gather_object([dataset])
    if accelerator.is_main_process:
        custom_dict = {}
        for dataset in gathered_dataset:
            for occupation, data in dataset.items():
                if occupation not in custom_dict:
                    custom_dict[occupation] = {"h_vects": [], "sensitive_prompts": [], "normal_prompts": [], "face_detected": [], "gender_probs": []}
                
                custom_dict[occupation]["h_vects"].extend(data["h_vects"])
                custom_dict[occupation]["sensitive_prompts"].extend(data["sensitive_prompts"])
                custom_dict[occupation]["normal_prompts"].extend(data["normal_prompts"])
                custom_dict[occupation]["face_detected"].extend(data["face_detected"])
                custom_dict[occupation]["gender_probs"].extend(data["gender_probs"])
        
        save_dataset(custom_dict, output_dir)

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.log("Dataset creation complete")

def create_argparser():
    defaults = dict(
        num_samples=100,  # Pro Occupation
        batch_size=20,
        use_fp16=True,
        seed="",
        output_dir="dataset",
        model_id="SG161222/Realistic_Vision_V2.0",
        plot_every=1  # Plotte Bilder nach jedem 5. Batch
    )
    parser = argparse.ArgumentParser()
    for k, v in defaults.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    return parser

if __name__ == "__main__":
    main()