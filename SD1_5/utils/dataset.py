import os
import torch as th
import csv
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import ToTensor

def read_csv_with_commas(file_path, occupation):
    with open(file_path, "r") as f:
        lines = f.readlines()

    prompts = []
    comma_count = occupation.count(",")
    for line in lines:
        parts = line.strip().split(",")
        if comma_count == 0:
            image_id, normal_prompt, sensitive_prompt = parts
        else:
            image_id = parts[0]
            normal_prompt = ",".join(parts[1:comma_count+2]).strip()
            sensitive_prompt = ",".join(parts[comma_count+2:]).strip()
        prompts.append((normal_prompt, sensitive_prompt))

    return prompts

class HVectsDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.metadata = []
        self.h_vects_dir = os.path.join(data_dir, 'h_vects')
        
        with open(os.path.join(data_dir, 'metadata.csv'), 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['face_detected'] == '1':
                    self.metadata.append(row)

    def __len__(self):
        return len(self.metadata) * 50  # 50 h_vects per image

    def __getitem__(self, idx):
        image_idx = idx // 50
        h_vect_idx = idx % 50
        
        metadata = self.metadata[image_idx]
        h_vects_file = os.path.join(self.h_vects_dir, metadata['h_vects_filename'])
        h_vects = np.load(h_vects_file)
        
        timestep = list(h_vects.keys())[h_vect_idx]
        h_vect = h_vects[timestep]
        
        gender_scores = [float(score) for score in metadata['gender_scores'].split(',')]
        
        return {
            'h_vect': th.from_numpy(h_vect).float(),
            'timestep': th.tensor(int(timestep)).long(),
            'gender_scores': th.tensor(gender_scores).float()
        }


class MinorityScoreDataset(Dataset):
    def __init__(self, data_dir, num_quantiles):
        self.data_dir = data_dir
        self.metadata = pd.read_csv(os.path.join(data_dir, "metadata.csv"), header=None, names=["idx", "prompt", "score"])
        self.num_quantiles = num_quantiles
        self.quantiles = pd.qcut(self.metadata["score"], q=num_quantiles, labels=False)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        image_path = os.path.join(self.data_dir, f"{self.metadata.iloc[idx]['idx']}.png")
        image = Image.open(image_path).convert("RGB")
        image_tensor = ToTensor()(image)
        quantile = self.quantiles[idx]
        return image_tensor, quantile


class OccupationDataset(Dataset):
    def __init__(self, data_dir, num_occupations=None):
        self.data_dir = data_dir
        self.occupations = [d for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d))][:num_occupations]
        self.image_files = []
        self.prompts = []
        self.indices = []

        for idx, occupation in enumerate(self.occupations):
            occ_dir = os.path.join(self.data_dir, occupation)
            image_files = sorted(f for f in os.listdir(occ_dir) if f.endswith(".png"))
            self.image_files.extend(os.path.join(occ_dir, f) for f in image_files)

            prompts = read_csv_with_commas(os.path.join(occ_dir, "prompts.csv"), occupation)
            self.prompts.extend(prompts)
            self.indices.extend(range(len(self.image_files) - len(image_files), len(self.image_files)))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        prompt, _ = self.prompts[idx]
        original_idx = self.indices[idx]
        image = Image.open(image_file).convert("RGB")
        return image, prompt, original_idx

    def select(self, indices):
        selected_dataset = OccupationDataset(self.data_dir)
        selected_dataset.image_files = [self.image_files[i] for i in indices]
        selected_dataset.prompts = [self.prompts[i] for i in indices]
        selected_dataset.indices = [self.indices[i] for i in indices]
        selected_dataset.original_indices = indices
        return selected_dataset

def collate_fn(batch):
    images = [item[0] for item in batch]
    prompts = [item[1] for item in batch]
    original_indices = [item[2] for item in batch]
    images = th.stack([ToTensor()(image) for image in images])
    return images, prompts, original_indices