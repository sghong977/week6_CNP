import torch
import torch.nn as nn
import torch.nn.functional as F

def NLLloss(y, mean, var):
    return torch.mean(torch.log(var)+ (y - mean)**2/(2*var))


# TODO -- Implement CNP model for 1d regression

class CNPs(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    # Output : mean, variance
    def forward(self, x_ctx, y_ctx, x_obs):
        pass

class Encoder(nn.Module):
    def __init__(self, x_dim=1, y_dim=1, r_dim=128):
        super().__init__()
        pass

    def forward(self, x, y):
        pass

class Decoder(nn.Module):
    def __init__(self, x_dim, r_dim, y_dim):
        super().__init__()
        pass

    def forward(self, x, r):
        pass
