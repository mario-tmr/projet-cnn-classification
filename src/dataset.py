"""
dataset.py

Dans cette partie, nous avons chargé le dataset MNIST et préparé les DataLoaders pour l'entraînement
et l'évaluation. 
"""

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def get_dataloaders(batch_size=64, data_root='./data'):


    # Transformations appliquées à chaque image :
    # Conversion en tenseur PyTorch (valeurs 0-1)
    # Normalisation avec la moyenne/écart-type standards de MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = torchvision.datasets.MNIST(
        root=data_root,
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = torchvision.datasets.MNIST(
        root=data_root,
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True   
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False 
    )

    return train_loader, test_loader


if __name__ == "__main__":
    
    train_loader, test_loader = get_dataloaders()
    print(f"Batchs d'entraînement : {len(train_loader)}")
    print(f"Batchs de test : {len(test_loader)}")
