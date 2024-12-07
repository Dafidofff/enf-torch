import math

import torch
import torch.nn as nn


class auto_decoder(nn.Module):

    def __init__(self, 
            num_samples: int,
            num_latents: int,
            latent_dim: int, 
            spatial_dim: int, 
            roto: bool = False
        ):
        super().__init__()
        
        # Initialise the positions
        lims = 1 - 1 / math.sqrt(num_latents)
        poses = torch.stack(torch.meshgrid(
                torch.linspace(-lims, lims, int(math.sqrt(num_latents))), 
                torch.linspace(-lims, lims, int(math.sqrt(num_latents)))
            ), axis=-1)
        poses = torch.reshape(poses, (1, -1, 2))
        poses = torch.repeat_interleave(poses, num_samples, 0)
        self.p = nn.Parameter(poses)

        # Initialise orientations (if roto=True)
        self.roto = roto
        if roto == True:
            ori = torch.linspace(0, 2*torch.pi, num_latents)
            ori = torch.reshape(ori, (1, -1, 1))
            ori = torch.repeat_interleave(ori, num_samples, 0)
            self.ori = nn.Parameter(ori)

        # Initialise the context vectors
        context_vecs = torch.ones((num_samples, num_latents, latent_dim))
        self.c = nn.Parameter(context_vecs)

        # Initialise the Gaussian window
        g = torch.ones((num_samples, num_latents, 1)) * 2 / math.sqrt(num_latents)
        self.g = nn.Parameter(g)

    def __call__(self, idx):
        if self.roto:
            return self.p[idx], self.c[idx], self.g[idx], self.ori[idx]
        else:
            return self.p[idx], self.c[idx], self.g[idx]




if __name__ == "__main__":
    ad = auto_decoder(
        num_samples=5, 
        num_latents=25, 
        latent_dim=25,
        spatial_dim=2,
        roto=True,
    )

    ad(2)