import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms

def visualize_dataset(dataloader, class_names, num_images: int = 16, mean = [0.485, 0.456, 0.406], 
                      std = [0.229, 0.224, 0.225]):
    """
    Visualiza uma grade de imagens do dataset com suas respectivas classes.

    Parâmetros:
    - dataloader: DataLoader a partir do qual extrair as imagens.
    - class_names (List[str]): Lista de nomes das classes.
    - num_images (int): Número de imagens a serem exibidas (deve ser um quadrado perfeito, como 16, 25, etc.).
    - mean (List[float]): Média usada para normalização das imagens.
    - std (List[float]): Desvio padrão usado para normalização das imagens.

    Retorna:
    - None
    """
    # Verifica se num_images é um quadrado perfeito
    grid_size = int(np.sqrt(num_images))
    if grid_size ** 2 != num_images:
        raise ValueError("num_images deve ser um quadrado perfeito (e.g., 16, 25, 36).")

    # Recupera um único lote de dados
    data_iter = iter(dataloader)
    images, labels = next(data_iter)

    # Seleciona o número desejado de imagens
    images = images[:num_images]
    labels = labels[:num_images]

    # Desnormaliza as imagens
    inv_normalize = transforms.Normalize(
        mean=[-m/s for m, s in zip(mean, std)],
        std=[1/s for s in std]
    )
    images = inv_normalize(images)

    # Clipa os valores para [0, 1]
    images = torch.clamp(images, 0, 1)

    # Cria uma grade de imagens
    grid_img = torchvision.utils.make_grid(images, nrow=grid_size)

    # Transforma para o formato H x W x C para exibição com matplotlib
    np_img = grid_img.numpy().transpose((1, 2, 0))

    # Plot
    plt.figure(figsize=(12, 12))
    plt.imshow(np_img)
    plt.axis('off')

    # Cria as legendas
    label_list = [class_names[label] if isinstance(label, int) else class_names[label.argmax()] for label in labels]
    
    # Adiciona as legendas abaixo das imagens
    plt.clf()
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    for idx in range(num_images):
        row = idx // grid_size
        col = idx % grid_size
        ax = axes[row, col]
        img = images[idx].numpy().transpose((1, 2, 0))
        ax.imshow(img)
        ax.set_title(label_list[idx], fontsize=10)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def visualize_mixed_dataset(images, labels, class_names, mean, std, num_images=16):
    """
    Visualiza imagens misturadas (por exemplo, após aplicação do Mixup) com suas respectivas classes.

    Parâmetros:
    - images: Tensor de imagens.
    - labels: Labels correspondentes (vetores de probabilidade, one-hot "soft").
    - class_names: Lista de nomes das classes.
    - mean: Média utilizada para normalização.
    - std: Desvio padrão utilizado para normalização.
    - num_images (int): Número de imagens a serem exibidas.
    """
    # Seleciona o número desejado de imagens
    images = images[:num_images]
    labels = labels[:num_images]

    # Desnormaliza as imagens
    inv_normalize = transforms.Normalize(
        mean=[-m/s for m, s in zip(mean, std)],
        std=[1/s for s in std]
    )
    images = inv_normalize(images)

    # Clipa os valores para [0, 1]
    images = torch.clamp(images, 0, 1)

    # Cria uma grade de imagens
    grid_size = int(np.sqrt(num_images))
    if grid_size ** 2 != num_images:
        raise ValueError("num_images deve ser um quadrado perfeito (e.g., 16, 25, 36).")

    plt.clf()
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    for idx in range(num_images):
        row = idx // grid_size
        col = idx % grid_size
        ax = axes[row, col]
        img = images[idx].cpu().numpy().transpose((1, 2, 0))

        # Seleciona a classe com maior valor no vetor one-hot
        class_idx = labels[idx].argmax().item()
        class_name = class_names[class_idx]

        ax.imshow(img)
        ax.set_title(class_name, fontsize=10)
        ax.axis('off')
    plt.tight_layout()
    plt.show()
