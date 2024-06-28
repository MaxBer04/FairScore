import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

TIMESTEPS = [980, 960, 940, 920, 900, 880, 860, 840, 820, 800, 780, 760, 740,
             720, 700, 680, 660, 640, 620, 600, 580, 560, 540, 520, 500, 480, 460,
             440, 420, 400, 380, 360, 340, 320, 300, 280, 260, 240, 220, 200, 180,
             160, 140, 120, 100, 80, 60, 40, 20, 0]

class GenderClassifier(nn.Module):
    def __init__(self, in_channels, image_size, out_channels, combine_vectors=False, prefix=None):
        super().__init__()
        self.combine_vectors = combine_vectors
        self.input_dim = in_channels * image_size * image_size
        
        self.linears = nn.ModuleList([nn.Linear(self.input_dim, out_channels) for _ in range(50)])
        self.prefix = prefix
        
    def forward(self, x, t):
        x = x.reshape(x.shape[0], -1)
        
        timestep_indices = torch.tensor([TIMESTEPS.index(ti.item()) for ti in t], device=x.device)                      
        batch_size = x.shape[0]
        
        selected_linears = torch.stack([self.linears[i].weight for i in timestep_indices])
        selected_biases = torch.stack([self.linears[i].bias for i in timestep_indices])
        
        output = torch.bmm(x.unsqueeze(1), selected_linears.transpose(1, 2)).squeeze(1) + selected_biases
        
        return output

class ResNet18GenderClassifier(nn.Module):
    def __init__(self, in_channels, image_size, out_channels, combine_vectors=False, prefix=None):
        super().__init__()
        self.combine_vectors = combine_vectors
        self.in_channels = in_channels
        self.image_size = image_size
        self.out_channels = out_channels
        self.prefix = prefix

        # Laden eines vortrainierten ResNet18
        self.resnet = models.resnet18(pretrained=True)
        
        # Anpassen des ersten Convolutional Layers für die Eingabedimensionen
        self.resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Ersetzen des letzten Fully Connected Layers
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()  # Entfernen des letzten Layers
        
        # Erstellen von 50 Fully Connected Layers für die Zeitschritte
        self.time_fcs = nn.ModuleList([nn.Linear(num_ftrs, out_channels) for _ in range(50)])

    def forward(self, x, t):
        # ResNet Feature Extraktion
        features = self.resnet(x)
        
        # Auswählen des richtigen FC Layers basierend auf dem Zeitschritt
        timestep_indices = torch.tensor([TIMESTEPS.index(ti.item()) for ti in t], device=x.device)
        batch_size = x.shape[0]
        
        selected_fcs = torch.stack([self.time_fcs[i].weight for i in timestep_indices])
        selected_biases = torch.stack([self.time_fcs[i].bias for i in timestep_indices])
        
        # Anwenden des ausgewählten FC Layers
        output = torch.bmm(features.unsqueeze(1), selected_fcs.transpose(1, 2)).squeeze(1) + selected_biases
        
        return output

def make_model(in_channels, image_size, out_channels, combine_vectors=False, prefix="train", model_type="linear"):
    if model_type == "linear":
        return GenderClassifier(in_channels, image_size, out_channels, combine_vectors, prefix)
    elif model_type == "resnet18":
        return ResNet18GenderClassifier(in_channels, image_size, out_channels, combine_vectors, prefix)
    else:
        raise ValueError("Ungültiger model_type. Wähle 'linear' oder 'resnet18'.")