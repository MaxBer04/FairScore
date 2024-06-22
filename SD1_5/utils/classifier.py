import torch
import torch.nn as nn
import torch.nn.functional as F

TIMESTEPS = [981, 961, 941, 921, 901, 881, 861, 841, 821, 801, 781, 761, 741,
             721, 701, 681, 661, 641, 621, 601, 581, 561, 541, 521, 501, 481, 461,
             441, 421, 401, 381, 361, 341, 321, 301, 281, 261, 241, 221, 201, 181,
             161, 141, 121, 101, 81, 61, 41, 21, 1]

class GenderClassifier(nn.Module):
    def __init__(self, in_channels, image_size, out_channels, combine_vectors=False):
        super().__init__()
        self.combine_vectors = combine_vectors
        if combine_vectors:
            self.input_dim = in_channels * image_size * image_size
        else:
            self.input_dim = (in_channels // 2) * image_size * image_size
        
        self.linears = nn.ModuleList([nn.Linear(self.input_dim, out_channels) for _ in range(50)])
        
    def forward(self, x, t):
        if not self.combine_vectors:
            #x = x[:, 1]  # Take only the second vector if not combining
            x = x[1].unsqueeze(0)
    
        x = x.reshape(x.shape[0], -1)
        
        # Handle batch of timesteps
        outputs = []
        for i in range(x.shape[0]):
            timestep = t[i].item()
            timestep_index = TIMESTEPS.index(timestep)
            output = self.linears[timestep_index](x[i].unsqueeze(0))
            outputs.append(output)
        
        return torch.cat(outputs, dim=0)

def make_model(in_channels, image_size, out_channels, combine_vectors=False):
    return GenderClassifier(in_channels, image_size, out_channels, combine_vectors)
