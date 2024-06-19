import torch as th
import torch.nn as nn
import torchvision
import numpy as np
from torchvision import transforms
from torchvision.transforms import ToPILImage

def load_fairface_model(device):
    model = torchvision.models.resnet34(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 18)
    model.load_state_dict(th.hub.load_state_dict_from_url('https://drive.google.com/uc?id=113QMzQzkBDmYMs9LwzvD-jxEZdBQ5J4X&export=download'))
    model = model.to(device)
    model.eval()
    return model

def predict_gender(face, fairface, device):
    if face is None:
        return []

    if not isinstance(face, th.Tensor):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    face_tensor = transform(face).to(device)

    output = fairface(face_tensor).cpu().detach().numpy()

    gender_output = output[:, 7:9]
    gender_scores = th.from_numpy(gender_output)
    gender_scores = th.softmax(gender_scores, dim=1)

    return gender_scores.tolist()[0]

def predict_race_gender(faces, fairface, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    face_tensors = th.stack([transform(ToPILImage()(face.squeeze(0))) for face in faces])
    face_tensors = face_tensors.to(device)

    outputs = fairface(face_tensors)
    outputs = outputs.cpu().detach().numpy()

    race_outputs = outputs[:, :7]
    gender_outputs = outputs[:, 7:9]

    race_scores = np.exp(race_outputs) / np.sum(np.exp(race_outputs), axis=1, keepdims=True)
    gender_scores = np.exp(gender_outputs) / np.sum(np.exp(gender_outputs), axis=1, keepdims=True)

    race_preds = np.argmax(race_scores, axis=1)
    gender_preds = np.argmax(gender_scores, axis=1)

    return race_preds, gender_preds, race_scores, gender_scores

def map_race(race_preds):
    race_map = {
        0: 'White',
        1: 'Black',
        2: 'Latino_Hispanic',
        3: 'East Asian',
        4: 'Southeast Asian',
        5: 'Indian',
        6: 'Middle Eastern'
    }
    return [race_map[pred] for pred in race_preds]

def map_gender(gender_preds):
    gender_map = {
        0: 'Male',
        1: 'Female'
    }
    return [gender_map[pred] for pred in gender_preds]

def map_race_4group(races):
    race_map = {
        'White': 'WMELH',
        'Middle Eastern': 'WMELH',
        'Latino_Hispanic': 'WMELH',
        'East Asian': 'Asian',
        'Southeast Asian': 'Asian',
        'Black': 'Black',
        'Indian': 'Indian'
    }
    return [race_map[race] for race in races]