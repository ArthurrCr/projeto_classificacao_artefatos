from pathlib import Path
from typing import List, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
from collections import Counter, defaultdict

class BalancedImageDataset(Dataset):
    def __init__(self, image_paths: List[str], labels: List[int], transform=None):
        """
        Dataset customizado para imagens balanceadas.

        Parâmetros:
        - image_paths (List[str]): Lista de caminhos das imagens.
        - labels (List[int]): Lista de labels correspondentes (em formato inteiro).
        - transform (callable, optional): Função de pré-processamento a ser aplicada nas imagens.
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        label = self.labels[idx]

        try:
            image = Image.open(path).convert('RGB')
        except Exception:
            # Retorna uma imagem preta caso a imagem não possa ser carregada
            image = Image.new('RGB', (256, 256), (0, 0, 0))
            print(f"Erro ao carregar a imagem: {path}")

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(label, dtype=torch.long)
        return image, label

def get_image_paths_and_labels(data_dir: str, classes: List[str]) -> Tuple[List[str], List[int]]:
    """
    Obtém os caminhos das imagens e suas respectivas labels a partir de um diretório de dados.
    Cada subpasta dentro de `data_dir` deve ter o nome de uma das classes.

    Parâmetros:
    - data_dir (str): Diretório onde estão armazenadas as imagens.
    - classes (List[str]): Lista de nomes das classes.

    Retorna:
    - image_paths (List[str]): Lista de caminhos completos para as imagens.
    - labels (List[int]): Lista de índices correspondentes às classes das imagens.
    """
    class_indices = {cls_name: idx for idx, cls_name in enumerate(classes)}
    image_paths = []
    labels = []
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
    data_dir = Path(data_dir)
    for class_name in classes:
        class_dir = data_dir / class_name
        class_idx = class_indices[class_name]
        if not class_dir.is_dir():
            continue
        for img_path in class_dir.iterdir():
            if img_path.is_file() and img_path.suffix.lower() in valid_extensions:
                image_paths.append(str(img_path))
                labels.append(class_idx)
    return image_paths, labels

def balance_dataset(image_paths: List[str], labels: List[int], classes: List[str]) -> Tuple[List[str], List[int]]:
    """
    Balanceia o conjunto de dados para que todas as classes tenham o mesmo número de amostras.

    Parâmetros:
    - image_paths (List[str]): Lista de caminhos das imagens.
    - labels (List[int]): Lista de labels correspondentes.
    - classes (List[str]): Lista de nomes das classes.

    Retorna:
    - balanced_image_paths (List[str]): Lista balanceada de caminhos das imagens.
    - balanced_labels (List[int]): Lista balanceada de labels.
    """
    class_to_images = {class_name: [] for class_name in classes}
    for path, label in zip(image_paths, labels):
        class_name = classes[label]
        class_to_images[class_name].append(path)

    # Seleciona a menor quantidade de amostras entre as classes
    min_count = min(len(imgs) for imgs in class_to_images.values())

    balanced_image_paths = []
    balanced_labels = []
    rng = np.random.default_rng(seed=42)

    for class_name in classes:
        images = class_to_images[class_name]
        class_idx = classes.index(class_name)
        if len(images) > min_count:
            images = rng.choice(images, size=min_count, replace=False)
        balanced_image_paths.extend(images)
        balanced_labels.extend([class_idx] * len(images))

    return balanced_image_paths, balanced_labels

def create_all_loaders(
    train_dir: str,
    test_dir: str,
    classes: List[str],
    preprocess_fn,
    batch_size: int = 32
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Cria os DataLoaders para treino, validação e teste combinando os dados das pastas de treino e teste.
    Os dados são unidos, balanceados e divididos de forma estratificada nas proporções de 
    70% para treino, 15% para validação e 15% para teste, garantindo que cada classe tenha o 
    mesmo número de amostras em cada conjunto.

    Parâmetros:
    - train_dir (str): Diretório de treino.
    - test_dir (str): Diretório de teste.
    - classes (List[str]): Lista dos nomes das classes.
    - preprocess_fn (callable): Função de pré-processamento das imagens.
    - batch_size (int): Tamanho do batch.

    Retorna:
    - train_loader (DataLoader): DataLoader do conjunto de treino.
    - val_loader (DataLoader): DataLoader do conjunto de validação.
    - test_loader (DataLoader): DataLoader do conjunto de teste.
    """
    # Obter os caminhos e labels de cada diretório
    image_paths_train, labels_train = get_image_paths_and_labels(train_dir, classes)
    image_paths_test, labels_test = get_image_paths_and_labels(test_dir, classes)

    # Combina os dados de treino e teste
    image_paths = image_paths_train + image_paths_test
    labels = labels_train + labels_test

    # Balanceia o conjunto para que cada classe tenha o mesmo número de amostras
    balanced_image_paths, balanced_labels = balance_dataset(image_paths, labels, classes)

    # Cria o dataset balanceado
    dataset = BalancedImageDataset(
        image_paths=balanced_image_paths,
        labels=balanced_labels,
        transform=preprocess_fn
    )

    # Divisão estratificada:
    # Para cada classe, separe os índices correspondentes e divida conforme as proporções desejadas.
    class_indices = defaultdict(list)
    for idx, label in enumerate(dataset.labels):
        label_int = int(label)
        class_indices[label_int].append(idx)
    
    train_indices = []
    val_indices = []
    test_indices = []
    
    train_ratio = 0.70
    val_ratio = 0.15
    rng = np.random.default_rng(seed=42)
    num_classes = len(classes)
    for cls in range(num_classes):
        indices = np.array(class_indices[cls])
        rng.shuffle(indices)
        total = len(indices)
        n_train = int(train_ratio * total)
        n_val = int(val_ratio * total)
        train_indices.extend(indices[:n_train])
        val_indices.extend(indices[n_train:n_train+n_val])
        test_indices.extend(indices[n_train+n_val:])
    
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)
    test_subset = Subset(dataset, test_indices)
    
    # Cria os DataLoaders
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

def print_dataset_stats(subset, subset_name: str, class_names: list):
    """
    Imprime o total de imagens e a quantidade por classe para um dataset (ou subset).
    
    Parâmetros:
    - subset: Objeto do tipo Subset.
    - subset_name (str): Nome do conjunto (ex.: 'Treino', 'Validação', 'Teste').
    - class_names (list): Lista com os nomes das classes.
    """
    original_dataset = subset.dataset
    labels = []
    for idx in subset.indices:
        label = original_dataset.labels[idx]
        labels.append(int(label))
    
    total = len(labels)
    counts = Counter(labels)
    
    print(f"{subset_name} - Total de imagens: {total}")
    for i, class_name in enumerate(class_names):
        print(f"  {class_name}: {counts[i]}")
