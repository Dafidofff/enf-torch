from collections import defaultdict
from argparse import Namespace
from tqdm.auto import trange
from pathlib import Path

import wandb
import torch
import torch.nn.functional as F

from enf.enf_simple import EquivariantNeuralField
from enf.autodecoder import AutoDecoder
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
    lr_enf = 1e-4,
    lr_latents = 5e-2,
    weight_decay = 0.0,
    val_ratio = 10,
    visualise_ratio = 5,
    log_wandb = False,
    save_im_local = False,
    checkpoint_path = None,
    save_path = "./checkpoints/",
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
if config.train.log_wandb:
    run = wandb.init(project="enf-min", entity="equivariance", job_type="train", config=config)

# Setup checkpoint dir
config.train.save_path = Path(config.train.save_path) / run.id if config.train.log_wandb else Path(config.train.save_path) / "debug"
config.train.save_path.mkdir(parents=True, exist_ok=True)

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
ad = AutoDecoder(
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

steps, best_val_loss = 0, torch.inf
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

        # Backward
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        steps += 1

        # Update loss
        train_loss += loss.item()

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
        epoch_val_loss = val_loss/ len(test_dloader)

        if config.train.log_wandb:
            wandb.log({"epoch_val_mse": epoch_val_loss})

        if epoch_val_loss < best_val_loss:
            torch.save(enf.state_dict(), config.train.save_path / "enf_params.pt")
            torch.save(AutoDecoder, config.train.save_path / "ad_params.pt")
            torch.save(config, config.train.save_path / "config.pt")

        if steps % config.train.visualise_ratio == 0:
            visualise_latents(
                enf, coords, z,
                img, img_shape, 
                save_to_disk=config.train.save_im_local,
                wandb_log=config.train.log_wandb
            )
    