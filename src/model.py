"""
model.py

Ici, on a défini l'architecture du CNN utilisé pour classifier les chiffres MNIST.
Aucun modèle pré-entraîné n'est utilisé : toutes les couches sont
définies manuellement à l'aide de nn.Module.

NB: On part sur un CNN simple pour la classification MNIST (0-9).
    Architecture : 2 blocs convolutionnels + 2 couches linéaires.
"""

import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):

    def __init__(self):
        super(SimpleCNN, self).__init__()

        # Bloc convolutionnel 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # Bloc convolutionnel 2
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(in_features=64 * 7 * 7, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

        # Dropout ici pour réduire le sur-apprentissage
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)          # [batch, 32, 14, 14]

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)          # [batch, 64, 7, 7]

        x = x.view(x.size(0), -1)  # Flatten -> [batch, 3136]

        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)            # Logits bruts, pas de softmax ici

        return x

from model import SimpleCNN
model = SimpleCNN()
print(model)