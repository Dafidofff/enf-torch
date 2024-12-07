from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10

class Cifar10WithID(Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform

        self.ds = CIFAR10(
            root=root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=True,
        )

    def __getitem__(self, index):
        img, target = self.ds[index]
        img = img.permute(1, 2, 0).float() / 255

        return img, target, index

    def __len__(self):
        return len(self.ds)
    

if __name__ == "__main__":
    ds = Cifar10WithID('./data')