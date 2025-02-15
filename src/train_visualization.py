import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import math
import torch
from sklearn.metrics import classification_report, confusion_matrix

def plot_loss_accuracy(train_losses, val_losses, train_accuracies, val_accuracies):
    """
    Plota os gráficos de Loss e Acurácia para treino e validação.
    
    Parâmetros:
    - train_losses: lista de perdas de treino por época.
    - val_losses: lista de perdas de validação por época.
    - train_accuracies: lista de acurácias de treino por época.
    - val_accuracies: lista de acurácias de validação por época.
    """
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(14, 5))
    
    # Plot da Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'bo-', label='Treino Loss')
    plt.plot(epochs, val_losses, 'ro-', label='Val Loss')
    plt.title('Loss por Época')
    plt.xlabel('Épocas')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot da Acurácia
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, 'bo-', label='Treino Acc')
    plt.plot(epochs, val_accuracies, 'ro-', label='Val Acc')
    plt.title('Acurácia por Época')
    plt.xlabel('Épocas')
    plt.ylabel('Acurácia')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def plot_classification_report(model, test_loader, classes, device=None):
    """
    Gera e imprime o classification_report.
    
    Parâmetros:
    - model: modelo PyTorch treinado.
    - test_loader: DataLoader do conjunto de teste.
    - classes: lista com o nome das classes.
    - device: dispositivo ('cuda' ou 'cpu'). Se None, é detectado automaticamente.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    y_true = []
    y_pred = []

    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    
    report = classification_report(y_true, y_pred, target_names=classes)
    print("Classification Report:")
    print(report)

def plot_confusion_matrix(model, test_loader, classes, device=None, normalize=False, cmap='Blues'):
    """
    Gera e plota a matriz de confusão.
    
    Parâmetros:
    - model: modelo PyTorch treinado.
    - test_loader: DataLoader do conjunto de teste.
    - classes: lista com o nome das classes.
    - device: dispositivo ('cuda' ou 'cpu'). Se None, é detectado automaticamente.
    - normalize: se True, normaliza a matriz de confusão.
    - cmap: mapa de cores para visualização.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    cm = confusion_matrix(all_labels, all_preds)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = 'Matriz de Confusão Normalizada'
    else:
        fmt = 'd'
        title = 'Matriz de Confusão'
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap=cmap, 
                xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.xlabel('Predição')
    plt.ylabel('Verdadeiro')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_wrong_predictions(model, dataloader, classes, mean, std, num_images=25, device=None):
    """
    Plota as imagens que o modelo classificou incorretamente.
    
    Parâmetros:
    - model: modelo PyTorch treinado.
    - dataloader: DataLoader do conjunto de teste.
    - classes: lista com os nomes das classes.
    - mean: lista com as médias utilizadas na normalização.
    - std: lista com os desvios padrão utilizados na normalização.
    - num_images: número máximo de imagens incorretas a serem exibidas.
    - device: dispositivo ('cuda' ou 'cpu'). Se None, é detectado automaticamente.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.to(device)
    model.eval()

    wrong_images = []
    true_labels = []
    pred_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            mask = preds != labels
            if mask.sum().item() > 0:
                wrong_images.extend(images[mask].cpu())
                true_labels.extend(labels[mask].cpu())
                pred_labels.extend(preds[mask].cpu())
            if len(wrong_images) >= num_images:
                break

    wrong_images = wrong_images[:num_images]
    true_labels = true_labels[:num_images]
    pred_labels = pred_labels[:num_images]

    def denormalize(img, mean, std):
        img = img.clone()
        for c in range(3):  # assumindo imagens RGB
            img[c] = img[c] * std[c] + mean[c]
        return img

    grid_size = math.ceil(math.sqrt(num_images))
    plt.figure(figsize=(15, 15))
    for idx in range(len(wrong_images)):
        img = denormalize(wrong_images[idx], mean, std)
        np_img = img.numpy().transpose((1, 2, 0))
        np_img = np.clip(np_img, 0, 1)
        plt.subplot(grid_size, grid_size, idx + 1)
        plt.imshow(np_img)
        plt.title(f"Verdadeiro: {classes[true_labels[idx]]}\nPredito: {classes[pred_labels[idx]]}", fontsize=8)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()
