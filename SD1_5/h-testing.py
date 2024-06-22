import torch as th 
from accelerate import Accelerator
from torchvision.transforms import ToTensor
from torchvision.utils import save_image
import numpy as np

from utils.custom_pipe import HDiffusionPipeline

def analyze_gender_predictions(probs_list, prompt_genders):
    # Konvertiere die Liste von Tensoren in einen einzigen Tensor
    probs = th.stack(probs_list)
    
    def calculate_accuracy(predictions, truths):
        correct = sum((pred == truth) for pred, truth in zip(predictions, truths))
        undecided = sum((pred == "undecided") for pred in predictions)
        return correct, undecided, len(truths)

    def get_prediction(male_prob, female_prob):
        if abs(male_prob - female_prob) < 0.1:  # 55% - 45% = 10%
            return "undecided"
        return "male" if male_prob > female_prob else "female"

    num_steps, num_images, _ = probs.shape
    
    # Gesamtanalyse
    avg_probs = probs.mean(dim=0)
    predictions = [get_prediction(male_prob, female_prob) for male_prob, female_prob in avg_probs]
    correct, undecided, total = calculate_accuracy(predictions, prompt_genders)

    print("\n--- Gesamtanalyse (alle Zeitschritte) ---")
    print(f"Korrekt: {correct}/{total} ({correct/total*100:.2f}%)")
    print(f"Unentschieden: {undecided}/{total} ({undecided/total*100:.2f}%)")
    
    for i, (pred, truth, prob) in enumerate(zip(predictions, prompt_genders, avg_probs)):
        print(f"Bild {i+1}: Vorhersage={pred}, {truth}, "
              f"m {prob[0]*100:.2f}% - w {prob[1]*100:.2f}%")

    # Analyse der ersten 15 Zeitschritte
    avg_probs_15 = probs[:15].mean(dim=0)
    predictions_15 = [get_prediction(male_prob, female_prob) for male_prob, female_prob in avg_probs_15]
    correct_15, undecided_15, _ = calculate_accuracy(predictions_15, prompt_genders)

    print("\n--- Analyse der ersten 15 Zeitschritte ---")
    print(f"Korrekt: {correct_15}/{total} ({correct_15/total*100:.2f}%)")
    print(f"Unentschieden: {undecided_15}/{total} ({undecided_15/total*100:.2f}%)")

    # Genauigkeit pro Zeitschritt
    accuracies = []
    for step in range(num_steps):
        step_preds = [get_prediction(male_prob, female_prob) for male_prob, female_prob in probs[step]]
        correct_step, _, _ = calculate_accuracy(step_preds, prompt_genders)
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
    pipe = HDiffusionPipeline.from_pretrained(model_id, torch_dtype=th.float16 if use_fp16 else th.float32)
    
    # Move the model to device before preparing
    pipe = pipe.to(accelerator.device)
    pipe = accelerator.prepare(pipe)
    pipe.init_classifier()
    
    m_mult = 8
    w_mult = 8
    
    images, h_vects, probs = pipe(["A photo of the face of a female dermatologist"]*w_mult+["A photo of the face of a male dermatologist"]*m_mult, num_inference_steps=50, guidance_scale=7.5, return_dict=False)

    # Umwandlung der Bilder in Tensoren
    tensor_images = [ToTensor()(img) for img in images]

    # Concatenation der Bilder entlang der Batch-Dimension (0-Achse)
    images = th.cat([img.unsqueeze(0) for img in tensor_images], dim=0)

    # Speichern des Bildes
    save_image(images, "test.png")
    
    # Analyse der Gendervorhersagen
    prompt_genders = ["female"] * w_mult + ["male"] * m_mult
    analyze_gender_predictions(probs, prompt_genders)
    
if __name__ == "__main__":
    main()