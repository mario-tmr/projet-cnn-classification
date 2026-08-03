"""
train.py

Dans cette partie, on entraîne le CNN sur le dataset MNIST.
On s'assure de sauvegarder à la fin :
  - les poids du modèle entraîné (outputs/mnist_cnn.pth)
  - l'historique des métriques (outputs/history.json), utilisé par evaluate.py
"""

import json
import os

import torch
import torch.nn as nn
import torch.optim as optim

from model import SimpleCNN
from dataset import get_dataloaders


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()  

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()          
        outputs = model(images)        
        loss = criterion(outputs, labels)  
        loss.backward()                
        optimizer.step()               

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total

"""
    Cette fonction elle, évalue le modèle sur un jeu de données, sans mettre à jour les poids.
    Elle va retourner (loss moyenne, accuracy).
    """

def evaluate(model, loader, criterion, device):
    model.eval() 

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total

# On teste ici si l GPU est disponible et on l'utilise. Sinon, on continue avec le CPU pour l'entrainement. 
def main(num_epochs=10, batch_size=64, lr=0.001, output_dir='outputs'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device utilisé : {device}")

    os.makedirs(output_dir, exist_ok=True)

    # Données
    train_loader, test_loader = get_dataloaders(batch_size=batch_size)

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Nombre de paramètres entraînables : {num_params:,}")

    # Entraînement
    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}

    print("\nDébut de l'entraînement...\n")
    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

    print("\nEntraînement terminé ")

    # Nos Sauvegardes
    model_path = os.path.join(output_dir, 'mnist_cnn.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Modèle sauvegardé : {model_path}")

    history_path = os.path.join(output_dir, 'history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Historique sauvegardé : {history_path}")


if __name__ == "__main__":
    main()
