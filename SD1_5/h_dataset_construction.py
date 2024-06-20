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
    
    # Speichere h_vects als NumPy-Arrays
    h_vects_dict = {t: [] for t in range(51)}
    for img_h_vects in dataset["h_vects"]:
        for t, h_vect in enumerate(img_h_vects):
            h_vects_dict[t].append(h_vect.cpu().numpy())
    for t, h_vects in h_vects_dict.items():
        np.savez(os.path.join(output_dir, f"h_vects_t{t}.npz"), h_vects=h_vects)

    # Speichere Metadaten in einer JSON-Datei
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump({
            "normal_prompts": dataset["normal_prompts"],
            "sensitive_prompts": dataset["sensitive_prompts"],
            "face_detected": dataset["face_detected"],
            "gender_probs": dataset["gender_probs"],
            "occupations": dataset["occupations"]
        }, f)

def plot_images(images, faces, boxes, gender_probs, batch_idx, output_dir, max_images):
    num_images = len(images)
    if max_images == 0:
        return
    if num_images > max_images:
        idxs = random.sample(range(num_images), max_images)
        images = [images[i] for i in idxs]
        faces = [faces[i] for i in idxs]
        boxes = [boxes[i] for i in idxs]
        gender_probs = [gender_probs[i] for i in idxs]

    size = int(np.sqrt(max_images))
    fig, axs = plt.subplots(size, size, figsize=(20, 8))
    axs = axs.ravel()

    for i, (img, face, box, gender_prob) in enumerate(zip(images[:max_images], faces[:max_images], boxes[:max_images], gender_probs[:max_images])):
        if not isinstance(img, th.Tensor):
            img = ToTensor()(img)
        img = img.cpu().numpy().transpose(1, 2, 0)
        img = (img * 255).astype(np.uint8)
        img = Image.fromarray(img)

        if face is not None:
            draw = ImageDraw.Draw(img)
            draw.rectangle(box.tolist(), outline="red", width=10)

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
        dataset = {"h_vects": [], "sensitive_prompts": [], "normal_prompts": [], "face_detected": [], "gender_probs": [], "occupations": []}

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

                # Speichere h_vects als Liste von Listen, wobei jede Unter-Liste die h_vects für ein Bild enthält
                h_vects_list = [[] for _ in range(batch_size)]
                for t in h_vects.keys():
                    for i in range(batch_size):
                        h_vects_list[i].append(h_vects[t][i*2:i*2+2])
                dataset["h_vects"].extend(h_vects_list)

                dataset["sensitive_prompts"].extend(sensitive_prompts)
                dataset["normal_prompts"].extend(normal_prompts)
                dataset["face_detected"].extend([face is not None for face in faces])
                dataset["gender_probs"].extend(gender_probs_batch)
                dataset["occupations"].extend([occupation] * batch_size)
                
                print(batch_size, accelerator.num_processes)
                progress_bar.update(batch_size*accelerator.num_processes)

                # Plotte Bilder nach jedem x-ten Batch
                if args.plot_every and (batch_start // args.batch_size) % args.plot_every == 0:
                    plot_images(images, faces, boxes, gender_probs_batch, batch_start // args.batch_size, output_dir, args.num_example_images)

    progress_bar.close()

    accelerator.wait_for_everyone()
    gathered_dataset = gather_object([dataset])
    if accelerator.is_main_process:
        custom_dict = {"h_vects": [], "sensitive_prompts": [], "normal_prompts": [], "face_detected": [], "gender_probs": [], "occupations": []}
        for dataset in gathered_dataset:
            custom_dict["h_vects"].extend(dataset["h_vects"])
            custom_dict["sensitive_prompts"].extend(dataset["sensitive_prompts"])
            custom_dict["normal_prompts"].extend(dataset["normal_prompts"])
            custom_dict["face_detected"].extend(dataset["face_detected"])
            custom_dict["gender_probs"].extend(dataset["gender_probs"])
            custom_dict["occupations"].extend(dataset["occupations"])
        
        save_dataset(custom_dict, output_dir)

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.log("Dataset creation complete")

def create_argparser():
    defaults = dict(
        num_samples=100,  # Pro Occupation
        batch_size=64,
        use_fp16=True,
        seed="",
        output_dir="dataset",
        model_id="SG161222/Realistic_Vision_V2.0",
        plot_every=5,  # per batch
        num_example_images=36 # should be square
    )
    parser = argparse.ArgumentParser()
    for k, v in defaults.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    return parser

if __name__ == "__main__":
    main()