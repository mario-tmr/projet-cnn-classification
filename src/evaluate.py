"""
evaluate.py

Cette partie charge le modèle entraîné et génère les
visualisations utiles:
  - courbes de perte/précision (à partir de outputs/history.json)
  - matrice de confusion
  - exemples de prédictions (bonnes et mauvaises)

Sauvegarde les figures dans outputs/.
"""

import json
import os

import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from model import SimpleCNN
from dataset import get_dataloaders


def plot_training_curves(history, output_dir):
    num_epochs = len(history['train_loss'])
    epochs_range = range(1, num_epochs + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs_range, history['train_loss'], label='Train Loss', marker='o')
    axes[0].plot(epochs_range, history['test_loss'], label='Test Loss', marker='o')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Évolution de la perte')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(epochs_range, history['train_acc'], label='Train Accuracy', marker='o')
    axes[1].plot(epochs_range, history['test_acc'], label='Test Accuracy', marker='o')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Évolution de la précision')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Courbes sauvegardées : {save_path}")


def plot_confusion_matrix(model, test_loader, device, output_dir):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Prédiction')
    plt.ylabel('Vraie valeur')
    plt.title('Matrice de confusion')

    save_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Matrice de confusion sauvegardée : {save_path}")


def plot_sample_predictions(model, test_loader, device, output_dir, num_samples=8):
    images, labels = next(iter(test_loader))
    images_device = images.to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(images_device)
        _, preds = torch.max(outputs, 1)
    preds = preds.cpu()

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for i, ax in enumerate(axes.flat):
        if i >= num_samples:
            break
        img = images[i][0] * 0.3081 + 0.1307  
        ax.imshow(img, cmap='gray')
        correct = preds[i].item() == labels[i].item()
        color = 'green' if correct else 'red'
        ax.set_title(f"Vrai: {labels[i].item()} | Prédit: {preds[i].item()}", color=color)
        ax.axis('off')
    plt.tight_layout()

    save_path = os.path.join(output_dir, 'sample_predictions.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Exemples de prédictions sauvegardés : {save_path}")


def main(output_dir='outputs'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device utilisé : {device}")

    # Chargeons ici le modèle entraîné
    model = SimpleCNN().to(device)
    model_path = os.path.join(output_dir, 'mnist_cnn.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Modèle chargé depuis : {model_path}")

    # Chargeons les données de test
    _, test_loader = get_dataloaders()

    # Historique d'entraînement
    history_path = os.path.join(output_dir, 'history.json')
    with open(history_path, 'r') as f:
        history = json.load(f)

    # Visualisations
    plot_training_curves(history, output_dir)
    plot_confusion_matrix(model, test_loader, device, output_dir)
    plot_sample_predictions(model, test_loader, device, output_dir)


if __name__ == "__main__":
    main()
