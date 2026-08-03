"""
dataset.py

Charge le dataset MNIST et prépare les DataLoaders pour l'entraînement
et l'évaluation. Utilise uniquement les données brutes MNIST (via
torchvision.datasets) : aucun modèle pré-entraîné n'est impliqué ici.
"""

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def get_dataloaders(batch_size=64, data_root='./data'):
    """
    Télécharger (si nécessaire) MNIST et retourner les DataLoaders
    d'entraînement et de test.
    """

    # Transformations appliquées à chaque image :
    # 1. Conversion en tenseur PyTorch (valeurs 0-1)
    # 2. Normalisation avec la moyenne/écart-type standards de MNIST
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
        shuffle=True   # Mélange à chaque epoch pour l'entraînement
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False  # Pas besoin de mélanger pour l'évaluation
    )

    return train_loader, test_loader


if __name__ == "__main__":
    # Petit test manuel : exécuter "python dataset.py" affiche un résumé
    train_loader, test_loader = get_dataloaders()
    print(f"Batchs d'entraînement : {len(train_loader)}")
    print(f"Batchs de test : {len(test_loader)}")
