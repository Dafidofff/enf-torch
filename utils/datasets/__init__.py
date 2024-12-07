import numpy as np
import torchvision
from torch.utils import data


def get_dataloader(cfg):
    if cfg.name == "stl10":
        train_dset = torchvision.datasets.STL10(root=cfg.path, split='train+unlabeled')
        test_dset = torchvision.datasets.STL10(root=cfg.path, split='test')
    elif cfg.name == "cifar10":
        from utils.datasets.cifar10_with_id import Cifar10WithID
        
        transform = torchvision.transforms.Compose([torchvision.transforms.PILToTensor()])
        train_dset = Cifar10WithID(root=cfg.path, train=True, transform=transform, target_transform=None)
        test_dset = Cifar10WithID(root=cfg.path, train=False, transform=transform, target_transform=None)
    else:
        raise ValueError(f"Unknown dataset name: {cfg.name}")

    if cfg.num_signals_train != -1:
        train_dset = data.Subset(train_dset, np.arange(0, cfg.num_signals_train))
    if cfg.num_signals_test != -1:
        test_dset = data.Subset(test_dset, np.arange(0, cfg.num_signals_test))

    train_loader = data.DataLoader(
        train_dset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        persistent_workers=False,
        drop_last=True
    )

    test_loader = data.DataLoader(
        test_dset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        persistent_workers=False,
        drop_last=True
    )

    return train_loader, test_loader
