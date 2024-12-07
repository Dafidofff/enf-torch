from collections import defaultdict
from argparse import Namespace
from tqdm.auto import trange

import wandb
import torch
import torch.nn.functional as F

from enf.enf_simple import EquivariantNeuralField
from enf.autodecoder import auto_decoder
from utils.datasets import get_dataloader
from utils.visualise import visualise_latents


# Define the config for train and model
# TODO: Replace by ML-collections
config = defaultdict(dict)
data_ns = Namespace(
    name = "cifar10",               # cifar10, stl10
    path = "./data",
    spatial_dim = 2,
    signal_dim = 3,
    num_signals_train = 128,
    num_signals_test = 32,
    batch_size = 32,
    num_workers = 0,
)
train_ns = Namespace(
    num_epochs = 1000,
    lr_enf = 1e-5,
    lr_latents = 5e-2,
    weight_decay = 0.0,
    val_ratio = 10,
    visualise_ratio = 5,
    log_wandb = True,
    save_im_local = False,
)
enf_ns = Namespace(
    num_latents = 25,
    latent_dim = 32,
    num_hidden = 128,
    att_dim = 128,
    num_heads = 3,
    emb_freq_q = 10,
    emb_freq_v = 20,
    nearest_k = 4
)

config = Namespace(
    data=data_ns,
    train=train_ns,
    enf=enf_ns
)

############################################
# Initialise all building blocks
############################################

# Initialize wandb
run = wandb.init(project="enf-min", entity="equivariance", job_type="train", config=config)

# Dataset
train_dloader, test_dloader = get_dataloader(config.data)

# Setup coords
smp = next(iter(train_dloader))[0]
img_shape = smp.shape[1:]
coords = torch.stack(torch.meshgrid(
        torch.linspace(-1, 1, smp.shape[1]), 
        torch.linspace(-1, 1, smp.shape[2]),
    ), axis=-1)
coords = torch.reshape(coords, (-1, 2))
coords = torch.repeat_interleave(coords[None, ...], config.data.batch_size, dim=0)

# Autodecoder
ad = auto_decoder(
    num_samples=config.data.num_signals_train + config.data.num_signals_test,
    num_latents=config.enf.num_latents,
    latent_dim=config.enf.latent_dim,
    spatial_dim=2,
)

# Neural Field
enf = EquivariantNeuralField(
    num_in=config.data.spatial_dim,
    num_out=config.data.signal_dim,
    latent_dim=config.enf.latent_dim, 
    num_hidden=config.enf.num_hidden,
    att_dim=config.enf.att_dim,
    num_heads=config.enf.num_heads,
    emb_freqs=(
        config.enf.emb_freq_q, 
        config.enf.emb_freq_v
    ),
    nearest_k=config.enf.nearest_k,
)

# Optimizer
optimizer = torch.optim.Adam([
    {'params': enf.parameters(), 'lr': config.train.lr_enf},
    {'params': ad.parameters(), 'lr': config.train.lr_latents},
], weight_decay=config.train.weight_decay)



############################################
# Training and validation loop
############################################

steps = 0
for i in (pbar := trange(config.train.num_epochs)):

    # Training epoch 
    train_loss = 0
    for batch in train_dloader:
        img, _, idx = batch
        img = img.reshape(config.data.batch_size, -1, config.data.signal_dim)

        # Get latents for samples and unpack
        z = (p, c, g) = ad(idx)

        # Reconstruct singal with latents
        out = enf(coords, p, c, g)

        # Calc loss
        loss = F.mse_loss(img, out, reduction='mean')
        train_loss += loss

        # Backward
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        steps += 1

        # Update pbar
        pbar.set_description(
              f"train step loss: {loss.item():.3f} | "
        )
    
    if config.train.log_wandb:
        wandb.log({"epoch_train_mse": train_loss / len(test_dloader)})


    if steps % config.train.val_ratio == 0:
        val_loss = 0
        for batch in test_dloader:
            img, _, idx = batch
            img = img.reshape(config.data.batch_size, -1, config.data.signal_dim)

            # Get latents for samples and unpack
            z = (p, c, g) = ad(idx)

            # Reconstruct singal with latents
            out = enf(coords, p, c, g)

            #  Calc loss
            loss = F.mse_loss(img, out, reduction='mean')
            val_loss += loss

            # Update pbar
            pbar.set_description(
              f"Val step loss: {loss.item():.3f} | "
            )

        if config.train.log_wandb:
            wandb.log({"epoch_val_mse": val_loss/ len(test_dloader)})

    if steps % config.train.visualise_ratio == 0:
        visualise_latents(
            enf, coords, z,
            img, img_shape, 
            save_to_disk=config.train.save_im_local,
            wandb_log=config.train.log_wandb
        )
    