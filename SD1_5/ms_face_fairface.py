import argparse
import os
import sys
import numpy as np
import torch as th
from torchvision.utils import save_image, make_grid
from torchvision.transforms import ToTensor
from tqdm import tqdm
from accelerate import Accelerator
from facenet_pytorch import MTCNN

# Importiere die benötigten Utility-Klassen und -Funktionen
from utils.dataset import OccupationDataset, collate_fn
from utils.face_detection import detect_faces
from utils.fairface import load_fairface_model, predict_gender
from utils.outputs import save_data
from utils.custom_pipe import CustomDiffusionPipeline

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, project_root)

def compute_minority_scores(args, global_dataset, local_dataset, pipe, fairface, accelerator):
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

            batch_gender_scores = [[] for _ in range(len(images))]
            if args.visual_check_interval:
                reconstructed_images = th.zeros(len(images) * (args.n_iter + 1), 3, images.shape[-2], images.shape[-1], device=accelerator.device)
                reconstructed_images[:len(images)] = images

            latents = pipe.vae.encode(images).latent_dist.sample().detach()
            latents = latents * pipe.vae.config.scaling_factor

            for i in range(args.n_iter):
                timestep = int(args.T_frac * pipe.scheduler.config.num_train_timesteps)
                timesteps = th.tensor([timestep] * len(images), dtype=th.long, device=accelerator.device)
                noise = th.randn_like(latents)
                noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

                denoised_images = pipe(prompts, latents=noisy_latents, t_frac=args.T_frac).images
                denoised_images = th.stack([ToTensor()(img) for img in denoised_images]).to(accelerator.device)
                denoised_images = (denoised_images * 2 - 1).clamp_(0.0, 1.0)

                # Erkenne Gesichter in den rekonstruierten Bildern
                denoised_faces, _ = detect_faces(denoised_images, mtcnn)

                # Berechne die Geschlechts-Scores für jedes Bild mit erkanntem Gesicht
                for j, (original_face, denoised_face) in enumerate(zip(original_faces, denoised_faces)):
                    if denoised_face is not None:
                        original_gender_scores = predict_gender(original_face.unsqueeze(0), fairface, accelerator.device)
                        denoised_gender_scores = predict_gender(denoised_face.unsqueeze(0), fairface, accelerator.device)
                        gender_score_distance = th.norm(original_gender_scores - denoised_gender_scores, p=2).item()
                        batch_gender_scores[j].append(gender_score_distance)

                if args.visual_check_interval:
                    reconstructed_images[len(images) * (i + 1):len(images) * (i + 2)] = denoised_images

            # Filtere Bilder ohne Scores heraus und berechne den Mittelwert der Scores für jedes Bild
            filtered_indices = [i for i, scores in enumerate(batch_gender_scores) if len(scores) > 0]
            filtered_prompts = [prompts[i] for i in filtered_indices]
            filtered_original_indices = [original_indices[i] for i in filtered_indices]
            filtered_batch_gender_scores = [np.mean(scores) for scores in batch_gender_scores if len(scores) > 0]

            # Speichere die ursprünglichen Indizes zusammen mit Prompts und Scores
            for idx, (prompt, score, original_idx) in enumerate(zip(filtered_prompts, filtered_batch_gender_scores, filtered_original_indices)):
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

def main():
    args = create_argparser().parse_args()

    accelerator = Accelerator()
    accelerator.print(f"Using {accelerator.num_processes} GPUs.")

    fairface = load_fairface_model(accelerator.device)

    accelerator.print("Loading Stable Diffusion model...")
    model_id = args.model_id
    pipe = CustomDiffusionPipeline.from_pretrained(model_id, torch_dtype=th.float16 if args.use_fp16 else th.float32)

    pipe = pipe.to(accelerator.device)
    if not accelerator.is_main_process:
        pipe.set_progress_bar_config(disable=True)

    accelerator.print("Creating dataset...")
    args.data_dir = os.path.join(script_dir, args.data_dir)
    dataset = OccupationDataset(args.data_dir, num_occupations=args.num_occupations)
    accelerator.print(f"Total of {len(dataset)} images")

    if accelerator.is_main_process:
        os.makedirs(os.path.join(script_dir, args.output_dir), exist_ok=True)
        if args.visual_check_interval:
            os.makedirs(os.path.join(script_dir, args.output_dir, 'reconstructed'), exist_ok=True)

    accelerator.wait_for_everyone()
    accelerator.print("Computing minority scores and saving reconstructions...")
    with accelerator.split_between_processes(list(range(len(dataset)))) as dataset_idcs:
        local_dataset = dataset.select(dataset_idcs)

        print(f"GPU{accelerator.process_index} working on {len(local_dataset)} entries")
        ms_tuples = compute_minority_scores(args, dataset, local_dataset, pipe, fairface, accelerator)

    save_data(args, dataset, ms_tuples, accelerator)

    accelerator.wait_for_everyone()
    print("Dataset construction complete")

def create_argparser():
    defaults = dict(
        batch_size=20,
        use_fp16=True,
        data_dir="dataset",
        output_dir="output-FF-08T-5it",
        model_id="SG161222/Realistic_Vision_V2.0",
        ms_compute_only=False,
        n_iter=5,
        visual_check_interval=None,
        num_occupations=None,
        save_interval=1, 
        T_frac=0.8,
    )
    parser = argparse.ArgumentParser()
    for k, v in defaults.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    return parser

if __name__ == "__main__":
    main()