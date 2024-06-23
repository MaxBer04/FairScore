import torch
import torch.nn as nn
import torch.nn.functional as F

TIMESTEPS = [981, 961, 941, 921, 901, 881, 861, 841, 821, 801, 781, 761, 741,
             721, 701, 681, 661, 641, 621, 601, 581, 561, 541, 521, 501, 481, 461,
             441, 421, 401, 381, 361, 341, 321, 301, 281, 261, 241, 221, 201, 181,
             161, 141, 121, 101, 81, 61, 41, 21, 1]

class GenderClassifier(nn.Module):
    def __init__(self, in_channels, image_size, out_channels, combine_vectors=False, prefix=None):
        super().__init__()
        self.combine_vectors = combine_vectors
        if combine_vectors:
            self.input_dim = in_channels * image_size * image_size
        else:
            self.input_dim = (in_channels // 2) * image_size * image_size
        
        self.linears = nn.ModuleList([nn.Linear(self.input_dim, out_channels) for _ in range(50)])
        self.prefix = prefix
        
    def forward(self, x, t):
        if not self.combine_vectors:
            x = x[:, 1]
        x = x.reshape(x.shape[0], -1)
        
        # Batch-Verarbeitung
        timestep_indices = torch.tensor([TIMESTEPS.index(ti.item()) for ti in t], device=x.device)
        batch_size = x.shape[0]
        
        # Wählen Sie die entsprechenden linearen Layer für jeden Zeitschritt aus
        selected_linears = torch.stack([self.linears[i].weight for i in timestep_indices])
        selected_biases = torch.stack([self.linears[i].bias for i in timestep_indices])
        
        # Führen Sie die Batch-Matrix-Multiplikation durch
        output = torch.bmm(x.unsqueeze(1), selected_linears.transpose(1, 2)).squeeze(1) + selected_biases
        
        return output

def make_model(in_channels, image_size, out_channels, combine_vectors=False, prefix="train"):
    return GenderClassifier(in_channels, image_size, out_channels, combine_vectors, prefix)