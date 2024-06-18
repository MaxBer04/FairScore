import os
import csv
from torchvision.utils import save_image
from torchvision.transforms import ToTensor
from accelerate.utils import gather_object

script_dir = os.path.dirname(os.path.abspath(__file__))

def save_data(output_path, dataset, ms_tuples, accelerator):
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
        images = [dataset[dataset.original_indices.index(idx)][0] for idx in indices]

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