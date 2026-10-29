import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

SEED = 68

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


def set_seed(seed=SEED):
    """Fija semilla individual para reproducibilidad."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_transforms():
    """
    Pipeline de transformaciones:
    - Aumento de datos: RandomCrop + RandomHorizontalFlip
    - Conversión a tensor
    - Normalización de canales RGB
    """
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    return train_transform, eval_transform


def get_dataloaders(
    data_dir="./data",
    batch_size=64,
    val_size=0.2,
    seed=SEED,
    num_workers=2
):
    """
    Carga CIFAR-10 y realiza partición estratificada train/val.
    Semilla individual: 68.
    """
    set_seed(seed)
    train_transform, eval_transform = get_transforms()

    train_full = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform
    )

    val_full = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=eval_transform
    )

    test_set = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=eval_transform
    )

    targets = np.array(train_full.targets)
    indices = np.arange(len(targets))

    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_size,
        stratify=targets,
        random_state=seed,
        shuffle=True
    )

    train_subset = Subset(train_full, train_idx)
    val_subset = Subset(val_full, val_idx)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    print("Partición estratificada con semilla:", seed)
    print(f"Train: {len(train_subset)} imágenes")
    print(f"Validación: {len(val_subset)} imágenes")
    print(f"Test: {len(test_set)} imágenes")

    return train_loader, val_loader, test_loader


def visualize_cifar10_samples(data_dir="./data", n=12, seed=SEED):
    """
    Muestra una cuadrícula con n imágenes de ejemplo del dataset CIFAR-10.
    n debe estar entre 8 y 16.
    """
    set_seed(seed)
    dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )

    indices = np.random.choice(len(dataset), n, replace=False)
    rows = 3 if n <= 12 else 4
    cols = 4
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
    axes = axes.flatten()

    for ax, idx in zip(axes, indices):
        image, label = dataset[idx]
        image = image.permute(1, 2, 0).numpy()
        ax.imshow(image)
        ax.set_title(CIFAR10_CLASSES[label])
        ax.axis("off")

    for ax in axes[len(indices):]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("outputs/samples.png", dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders(seed=SEED)
    visualize_cifar10_samples(n=12, seed=SEED)

    images, labels = next(iter(train_loader))
    print("Shape lote de imágenes:", images.shape)
    print("Shape lote de etiquetas:", labels.shape)