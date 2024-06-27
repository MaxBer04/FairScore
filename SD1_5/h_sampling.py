import torch as th 
from accelerate import Accelerator
from torchvision.transforms import ToTensor
from torchvision.utils import save_image
import numpy as np
from collections import Counter

from utils.custom_pipe import HDiffusionPipeline
from utils.semdiff import StableSemanticDiffusion, ConditionalUnet
from diffusers import StableDiffusionPipeline, DDIMScheduler

def analyze_gender_predictions(probs_list, prompt_genders):
    probs = th.stack(probs_list)
    
    def get_prediction(male_prob, female_prob):
        if abs(male_prob - female_prob) < 0.1:  # 55% - 45% = 10%
            return "undecided"
        return "male" if male_prob > female_prob else "female"

    num_steps, num_images, _ = probs.shape
    
    # Summe der Ergebnisse über alle Zeitschritte
    all_predictions = []
    for img in range(num_images):
        img_predictions = [get_prediction(male_prob, female_prob) for male_prob, female_prob in probs[:, img, :]]
        prediction_counts = Counter(img_predictions)
        all_predictions.append(prediction_counts)

    final_predictions = []
    for pred_count in all_predictions:
        if pred_count['male'] > pred_count['female']:
            final_predictions.append('male')
        elif pred_count['female'] > pred_count['male']:
            final_predictions.append('female')
        else:
            final_predictions.append('undecided')

    correct = sum((pred == truth) for pred, truth in zip(final_predictions, prompt_genders))
    undecided = sum((pred == "undecided") for pred in final_predictions)
    total = len(prompt_genders)

    print("\n--- Gesamtanalyse (Summe über alle Zeitschritte) ---")
    print(f"Korrekt: {correct}/{total} ({correct/total*100:.2f}%)")
    print(f"Unentschieden: {undecided}/{total} ({undecided/total*100:.2f}%)")
    
    for i, (pred, truth, counts) in enumerate(zip(final_predictions, prompt_genders, all_predictions)):
        print(f"Bild {i+1}: Vorhersage={pred}, {truth}, "
              f"m {counts['male']} - w {counts['female']} - u {counts['undecided']}")

    # Analyse der ersten 15 Zeitschritte
    all_predictions_15 = []
    for img in range(num_images):
        img_predictions = [get_prediction(male_prob, female_prob) for male_prob, female_prob in probs[15:, img, :]]
        prediction_counts = Counter(img_predictions)
        all_predictions_15.append(prediction_counts)

    final_predictions_15 = []
    for pred_count in all_predictions_15:
        if pred_count['male'] > pred_count['female']:
            final_predictions_15.append('male')
        elif pred_count['female'] > pred_count['male']:
            final_predictions_15.append('female')
        else:
            final_predictions_15.append('undecided')

    correct_15 = sum((pred == truth) for pred, truth in zip(final_predictions_15, prompt_genders))
    undecided_15 = sum((pred == "undecided") for pred in final_predictions_15)

    print("\n--- Analyse der letzten 15 Zeitschritte ---")
    print(f"Korrekt: {correct_15}/{total} ({correct_15/total*100:.2f}%)")
    print(f"Unentschieden: {undecided_15}/{total} ({undecided_15/total*100:.2f}%)")

    # Genauigkeit pro Zeitschritt
    accuracies = []
    for step in range(num_steps):
        step_preds = [get_prediction(male_prob, female_prob) for male_prob, female_prob in probs[step]]
        correct_step = sum((pred == truth) for pred, truth in zip(step_preds, prompt_genders))
        accuracies.append(correct_step / total * 100)

    print("\n--- Genauigkeit pro Zeitschritt ---")
    for step, acc in enumerate(accuracies):
        print(f"Schritt {step+1}: {acc:.2f}%")

    # Visualisierung der Genauigkeit über die Zeit
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, num_steps+1), accuracies)
        plt.title("Genauigkeit pro Zeitschritt")
        plt.xlabel("Zeitschritt")
        plt.ylabel("Genauigkeit (%)")
        plt.ylim(0, 100)
        plt.grid(True)
        plt.savefig("accuracy_over_time.png")
        print("\nDiagramm 'accuracy_over_time.png' wurde erstellt.")
    except ImportError:
        print("\nMatplotlib nicht installiert. Diagramm konnte nicht erstellt werden.")

def main():
    use_fp16 = False

    accelerator = Accelerator()
  
    model_id = "SG161222/Realistic_Vision_V2.0"
    #pipe = HDiffusionPipeline.from_pretrained(model_id, torch_dtype=th.float16 if use_fp16 else th.float32)
    pipe = StableDiffusionPipeline.from_pretrained(model_id).to(accelerator.device) 
    scheduler = pipe.scheduler

    pipe = StableSemanticDiffusion(
        unet=ConditionalUnet(pipe.unet),
        scheduler=DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False),
        vae = pipe.vae,
        tokenizer = pipe.tokenizer,
        text_encoder = pipe.text_encoder,
        model_id = model_id,
        num_inference_steps=50
    )
    
    outputs = pipe.sample(prompts = ["A photo of the face of a technical education teacher"]*2)
    save_image(outputs.x0, "x0.png")
    print(len(outputs.hs))
    print(outputs.hs[0].size())
    
    # Move the model to device before preparing
    #pipe = pipe.to(accelerator.device)
    #pipe = accelerator.prepare(pipe)
    #pipe.init_classifier('/root/FairScore/model_199.pt')
    
    #m_mult = 8
    #w_mult = 8

    # Umwandlung der Bilder in Tensoren
    #tensor_images = [ToTensor()(img) for img in images]

    # Concatenation der Bilder entlang der Batch-Dimension (0-Achse)
    #images = th.cat([img.unsqueeze(0) for img in tensor_images], dim=0)

    # Speichern des Bildes
    #save_image(images, "test.png")
    
    # Analyse der Gendervorhersagen
    #prompt_genders = ["female"] * w_mult + ["male"] * m_mult
    #analyze_gender_predictions(probs, prompt_genders)
    
if __name__ == "__main__":
    main()