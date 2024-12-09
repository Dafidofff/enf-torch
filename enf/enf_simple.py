from typing import Tuple

import torch
import torch.nn as nn


# class PosEmb(nn.Module):
#     num_freqs: int
#     freq: float

#     @nn.compact
#     def __call__(self, coords: jnp.ndarray) -> jnp.ndarray:
#         emb = nn.Dense(self.num_freqs // 2, kernel_init=nn.initializers.normal(self.freq), use_bias=False)(
#             jnp.pi * (coords + 1))  # scale to [0, 2pi]
#         return jnp.sin(jnp.concatenate([coords, emb, emb + jnp.pi / 2.0], axis=-1))
    

class SinusoidalPosEmbedding(nn.Module):

    def __init__(self, num_in, num_freqs: int, freq: float):
        super().__init__()
        self.num_freqs = num_freqs
        self.freq = freq

        # Set the embedding frequency.
        self.emb_layer = nn.Linear(in_features=num_in, out_features=self.num_freqs // 2, bias=False)
        nn.init.normal_(self.emb_layer.weight, std=self.freq)

    def __call__(self, coords: torch.Tensor) -> torch.Tensor:
        # Map coords to from [-1,1] to [0, 2pi]
        coords = torch.pi * (coords + 1)

        # Embed
        emb = self.emb_layer(coords)

        # Return sin and cosin
        return torch.sin(torch.concat([coords, emb, emb + torch.pi / 2.0], dim=-1))



class EquivariantNeuralField(nn.Module):
    """ Equivariant cross attention network for the latent points, conditioned on the poses.

    Args:
        num_hidden (int): The number of hidden units.
        att_dim (int): The dimensionality of the attention queries and keys.
        num_heads (int): The number of attention heads.
        num_out (int): The number of output coordinates.
        emb_freq (float): The frequency of the positional embedding.
        nearest_k (int): The number of nearest latents to consider.
    """
    def __init__(
        self,
        num_in: int,
        num_out: int,
        latent_dim: int, 
        num_hidden: int,
        att_dim: int,
        num_heads: int,
        emb_freqs: Tuple[float, float],
        nearest_k: int,
    ):
        super().__init__()

        # save params
        self.num_in = num_in
        self.num_out = num_out
        self.latent_dim = latent_dim
        self.num_hidden = num_hidden
        self.att_dim = att_dim
        self.num_heads = num_heads
        self.emb_freq_q = emb_freqs[0]
        self.emb_freq_v = emb_freqs[1]
        self.nearest_k = nearest_k

        # Stem maps latent to hidden space
        self.W_stem = nn.Linear(self.latent_dim, self.num_hidden)

        # Positional embedding, takes in relative positions.
        self.W_emb_q = nn.Sequential(
            SinusoidalPosEmbedding(self.num_in, self.num_hidden, self.emb_freq_q),
            nn.Linear(self.num_hidden+self.num_in, self.num_hidden), 
            nn.GELU(), 
            nn.Linear(self.num_hidden, self.num_heads * self.att_dim)
        )
        self.W_emb_v = nn.Sequential(
            SinusoidalPosEmbedding(self.num_in, self.num_hidden, self.emb_freq_v),
            nn.Linear(self.num_hidden+self.num_in, self.num_hidden), 
            nn.GELU(), 
            nn.Linear(self.num_hidden, 2 * self.num_hidden),
        )

        # Query, key, value functions.
        self.W_k = nn.Linear(self.num_hidden, self.num_heads * self.att_dim)
        self.W_v = nn.Linear(self.num_hidden, self.num_heads * self.num_hidden)
        self.softmax = nn.Softmax(dim=2)

        # Output layers.
        self.W_out = nn.Sequential(
            nn.Linear(self.num_heads * self.num_hidden, self.num_heads * self.num_hidden), 
            nn.GELU(), 
            nn.Linear(self.num_heads * self.num_hidden, self.num_out)
        )

    def __call__(self, x, p, c, g):
        """Apply equivariant cross attention.

        Args:
            x (torch.Tensor): The input coordinates. Shape (batch_size, num_coords, coord_dim).
            p (torch.Tensor): The latent poses. Shape (batch_size, num_latents, coord_dim).
            c (torch.Tensor): The latent context vectors. Shape (batch_size, num_latents, latent_dim).
            g (torch.Tensor): The window size for the gaussian window. Shape (batch_size, num_latents, 1).

        Returns:
            torch.Tensor: The output after applying the equivariant cross attention mechanism.
        """
        
        # Map latent to hidden space
        c = self.W_stem(c)

        # Calculate relative positions between input coordinates and latents.
        rel_pos = x[:, :, None, :] - p[:, None, :, :]

        # Calculate keys first, since they are not dependent on the input coordinates.
        k = self.W_k(c)

        # Take top-k nearest latents to every input coordinate.
        zx_dist = torch.sum(rel_pos ** 2, dim=-1)
        nearest_z = torch.argsort(zx_dist, dim=-1)[:, :, :self.nearest_k, None]

        # Restrict the relative positions and context vectors to the nearest latents.
        zx_dist = torch.take_along_dim(zx_dist[..., None], nearest_z, dim=2)
        rel_pos = torch.take_along_dim(rel_pos, nearest_z, dim=2)
        c = torch.take_along_dim(c[:, None, :, :], nearest_z, dim=2)
        k = torch.take_along_dim(k[:, None, :, :], nearest_z, dim=2)
        g = torch.take_along_dim(g[:, None, :, :], nearest_z, dim=2)

        # Apply positional embedding to get query and value conditioning.
        q = self.W_emb_q(rel_pos)
        v_g, v_b = torch.split(self.W_emb_v(rel_pos), split_size_or_sections=self.num_hidden, dim=-1)
        
        # Get value from conditioned context vectors.
        v = self.W_v(c * v_g + v_b)

        # Reshape to separate the heads
        q = q.reshape(q.shape[:-1] + (self.num_heads, -1))
        k = k.reshape(k.shape[:-1] + (self.num_heads, -1))
        v = v.reshape(v.shape[:-1] + (self.num_heads, -1))

        # Calculate the attention weights, apply gaussian mask based on distance, broadcasting over heads.
        att_logits = (q * k).sum(axis=-1, keepdims=True) - ((1 / g ** 2) * zx_dist)[..., None, :]
        att = self.softmax(att_logits)

        # Attend the values to the queries and keys.
        y = (att * v).sum(axis=2)

        # Combine the heads and apply the output layers.
        y = y.reshape(y.shape[:-2] + (-1,))
        return self.W_out(y)

