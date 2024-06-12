import argparse
import os
import sys
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from torchvision.utils import make_grid, save_image
from torchvision.transforms import ToTensor
from math import ceil, sqrt

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, project_root)

def load_image(data_dir, idx):
    image_file = os.path.join(data_dir, f"{idx}.png")
    return Image.open(image_file)

def add_score_to_image(image, score):
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 48)
    text = f"Score: {score:.4f}"
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    text_position = ((image.width - text_width) // 2, (image.height - text_height) // 2)
    draw.text(text_position, text, (255, 0, 0), font=font)
    return image

def create_image_grid(args):
    # Lade die Metadaten aus der CSV-Datei ohne Header
    metadata_file = os.path.join(args.data_dir, "metadata.csv")
    metadata = pd.read_csv(metadata_file, header=None)
    
    # Benenne die Spalten entsprechend ihrer Bedeutung
    metadata.columns = ["idx", "prompt", "score"]

    # Sortiere die Metadaten nach dem Minority Score
    metadata = metadata.sort_values("score")

    # Teile die Daten in Quantile auf
    quantiles = np.array_split(metadata, args.num_quantiles)

    # Berechne die Anzahl der Reihen und Spalten für die Image Grids
    num_images = args.num_images_per_grid
    num_cols = int(ceil(sqrt(num_images)))
    num_rows = int(ceil(num_images / num_cols))

    # Erstelle die Image Grids für die Quantile
    image_grids = []
    for quantile in quantiles:
        # Lade die Bilder für das aktuelle Quantil
        images = [load_image(args.data_dir, idx) for idx in quantile["idx"][:num_images]]
        
        # Füge die Scores zu den Bildern hinzu
        images_with_scores = [add_score_to_image(image, score) for image, score in zip(images, quantile["score"][:num_images])]
        
        # Konvertiere die PIL-Bilder in Tensoren
        tensor_images = [ToTensor()(image) for image in images_with_scores]
        
        # Erstelle das Image Grid für das aktuelle Quantil
        grid = make_grid(tensor_images, nrow=num_cols, padding=20)
        image_grids.append(grid)

    # Kombiniere die Image Grids der Quantile mit ausreichend Platz dazwischen
    combined_grid = make_grid(image_grids, nrow=1, padding=50)

    # Speichere das kombinierte Image Grid
    output_file = os.path.join(args.data_dir, "image_grid.png")
    save_image(combined_grid, output_file)

def main():
    args = create_argparser().parse_args()
    args.data_dir = os.path.join(script_dir, args.data_dir)
    print(f"Creating image grid from data in {args.data_dir}")
    create_image_grid(args)
    print("Image grid creation complete")

def create_argparser():
    defaults = dict(
        data_dir="dataset_2_ms",
        num_quantiles=4,
        num_images_per_grid=40,
    )
    parser = argparse.ArgumentParser()
    for k, v in defaults.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    return parser

if __name__ == "__main__":
    main()