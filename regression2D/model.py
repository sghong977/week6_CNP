import torch
import torch.nn as nn
import torch.nn.functional as F

#------------------------ MODEL --------------------------------------

def NLLloss(y, mean, var):
    return torch.mean(torch.log(var)+ (y - mean)**2/(2*var))

class CNPs(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x_ctx, y_ctx, x_obs):
        r_context = self.encoder(x_ctx, y_ctx)
        mean, variance = self.decoder(x_obs, r_context)

        return mean, variance

class Encoder(nn.Module):
    def __init__(self, x_dim=1, y_dim=1, r_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(x_dim + y_dim, 32)
        self.fc2 = nn.Linear(32, r_dim)

    def forward(self, x, y):
        r = torch.cat([x, y], dim=1)
        
        r = F.relu(self.fc1(r), inplace=True)
        r = F.relu(self.fc2(r), inplace=True)
        
        aggr_r = torch.mean(r, dim=0)   # aggregator
        return aggr_r

class Decoder(nn.Module):
    def __init__(self, x_dim, r_dim, y_dim):
        super().__init__()
        self.fc1 = nn.Linear(x_dim + r_dim, 32)
        self.fc2 = nn.Linear(32, 32)
        self.r_dim = r_dim
        
        self.fc_mu = nn.Linear(32, y_dim)
        self.fc_var = nn.Linear(32, y_dim)

    def forward(self, x, r):
        r = r.repeat(len(x))
        r = r.reshape(len(x), self.r_dim)
        y = torch.cat([x, r], dim=-1)
        
        y = F.relu(self.fc1(y), inplace=True)
        y = F.relu(self.fc2(y), inplace=True)
        
        mu = self.fc_mu(y)

        var = self.fc_var(y)
        var = F.softplus(var) + 10**-6
        
        return mu, var
