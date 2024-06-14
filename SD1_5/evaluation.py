import argparse
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageDraw
import torch
from facenet_pytorch import MTCNN
from torchvision.utils import make_grid, save_image

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, project_root)

from utils.fairface import load_fairface_model, predict_race_gender, map_race, map_gender, map_race_4group
from utils.visualization import visualize_faces

def load_image(data_dir, idx):
    image_file = os.path.join(data_dir, f"{idx}.png")
    return Image.open(image_file)

def detect_faces(image, mtcnn):
    # Konvertiere das Bild in ein RGB-Bild
    image = image.convert('RGB')

    # Erkenne Gesichter im Bild und erhalte die Bounding Boxes
    boxes, _ = mtcnn.detect(image)

    # Extrahiere die erkannten Gesichter aus dem Bild
    faces = mtcnn(image)

    return faces, boxes

def analyze_images(args, metadata, quantile=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    mtcnn = MTCNN(keep_all=True, device=device)
    fairface = load_fairface_model(device)

    race_data = []
    gender_data = []
    race_4group_data = []

    if args.quantiles > 1:
        metadata = metadata[metadata['quantile'] == quantile]

    print(f"Analyzing images for quantile {quantile if quantile is not None else 'all'}...")
    for batch_start in range(0, len(metadata), args.batch_size):
        batch_end = min(batch_start + args.batch_size, len(metadata))
        batch_metadata = metadata.iloc[batch_start:batch_end]

        batch_images = []
        batch_boxes = []
        if args.quantiles > 1:
            for idx, prompt, score, quantile in batch_metadata.values:
                image = load_image(args.data_dir, idx)
                faces, boxes = detect_faces(image, mtcnn)
                if faces is not None:
                    batch_images.extend(faces)
                    batch_boxes.extend(boxes)
        else:
            for idx, prompt, score in batch_metadata.values:
                image = load_image(args.data_dir, idx)
                faces, boxes = detect_faces(image, mtcnn)
                if faces is not None:
                    batch_images.extend(faces)
                    batch_boxes.extend(boxes)

        if len(batch_images) > 0:
            race_preds, gender_preds, _, _ = predict_race_gender(batch_images, fairface)
            races = map_race(race_preds)
            genders = map_gender(gender_preds)
            race_4groups = map_race_4group(races)

            race_data.extend(races)
            gender_data.extend(genders)
            race_4group_data.extend(race_4groups)

            if (batch_start // args.batch_size) % args.visualization_steps == 0:
                visualize_faces(args, batch_images, batch_boxes, batch_start // args.batch_size, quantile)

    return race_data, gender_data, race_4group_data

def plot_distribution(args, data, label, quantile=None):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.countplot(x=data, ax=ax)
    ax.set_title(f"{label} Distribution{' - Quantile ' + str(quantile) if quantile is not None else ''}")
    ax.set_xlabel(label)
    ax.set_ylabel('Count')
    plt.tight_layout()

    if args.quantiles > 1:
        output_file = os.path.join(args.output_dir, f"{label.lower()}_distribution_quantile_{quantile}.png")
    else:
        output_file = os.path.join(args.output_dir, f"{label.lower()}_distribution.png")

    plt.savefig(output_file)
    plt.close()
    print(f"{label} distribution plot saved at {output_file}")

def main():
    args = create_argparser().parse_args()
    args.data_dir = os.path.join(script_dir, args.data_dir)
    args.output_dir = os.path.join(script_dir, args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading metadata...")
    metadata_file = os.path.join(args.data_dir, "metadata.csv")
    metadata = pd.read_csv(metadata_file, header=None, names=['idx', 'prompt', 'score'])

    if args.quantiles > 1:
        print(f"Splitting data into {args.quantiles} quantiles...")
        metadata['quantile'] = pd.qcut(metadata['score'], q=args.quantiles, labels=False)

        for quantile in range(args.quantiles):
            print(f"Processing quantile {quantile + 1}/{args.quantiles}")
            race_data, gender_data, race_4group_data = analyze_images(args, metadata, quantile)
            plot_distribution(args, race_data, 'Race', quantile)
            plot_distribution(args, gender_data, 'Gender', quantile)
            plot_distribution(args, race_4group_data, 'Race (4 Groups)', quantile)
    else:
        print("Processing all data...")
        race_data, gender_data, race_4group_data = analyze_images(args, metadata)
        plot_distribution(args, race_data, 'Race')
        plot_distribution(args, gender_data, 'Gender')
        plot_distribution(args, race_4group_data, 'Race (4 Groups)')

    print("Analysis complete.")

def create_argparser():
    defaults = dict(
        data_dir="dataset_2_ms_faces",
        output_dir="ms_with_faces_analysis/q-4",
        quantiles=4,
        batch_size=640,
        visualization_steps=2,
    )
    parser = argparse.ArgumentParser()
    for k, v in defaults.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    return parser

if __name__ == "__main__":
    main()