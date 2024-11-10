import os
os.environ['PYTHONHASHSEED'] = '42'

import random
random.seed(42)
import tensorflow as tf
tf.random.set_seed(42)
from collections import Counter
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess
from tensorflow.keras.applications.vgg19 import preprocess_input as vgg19_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenetv2_preprocess
import numpy as np
np.random.seed(42)



def get_image_paths_and_labels(data_dir, classes):
    """
    Obtém os caminhos das imagens e suas respectivas labels a partir de um diretório de dados.

    Parâmetros:
    - data_dir (str): Diretório onde estão armazenadas as imagens.
    - classes (list): Lista de nomes das classes.

    Retorna:
    - image_paths (list): Lista de caminhos completos para as imagens.
    - labels (list): Lista de índices correspondentes às classes das imagens.
    """
    class_indices = {cls_name: idx for idx, cls_name in enumerate(classes)}
    image_paths = []
    labels = []
    for cls in classes:
        cls_dir = os.path.join(data_dir, cls)
        cls_index = class_indices[cls]
        if not os.path.isdir(cls_dir):
            print(f"Diretório para a classe '{cls}' não encontrado em {data_dir}. Pulando...")
            continue
        for img_name in os.listdir(cls_dir):
            img_path = os.path.join(cls_dir, img_name)
            if os.path.isfile(img_path):
                image_paths.append(img_path)
                labels.append(cls_index)
    return image_paths, labels


def balance_dataset(image_paths, labels, classes):
    """
    Balanceia o conjunto de dados para que todas as classes tenham o mesmo número de amostras.

    Parâmetros:
    - image_paths (list): Lista de caminhos das imagens.
    - labels (list): Lista de labels correspondentes.
    - classes (list): Lista de nomes das classes.

    Retorna:
    - balanced_image_paths (list): Lista balanceada de caminhos das imagens.
    - balanced_labels (list): Lista balanceada de labels.
    """
    class_to_images = {cls: [] for cls in classes}
    class_indices = {cls_name: idx for idx, cls_name in enumerate(classes)}
    for path, label in zip(image_paths, labels):
        cls = classes[label]
        class_to_images[cls].append(path)
    
    min_count = min(len(imgs) for imgs in class_to_images.values())
    print(f"Número mínimo de imagens por classe para balanceamento: {min_count}")
    
    balanced_image_paths = []
    balanced_labels = []
    
    for cls in classes:
        imgs = class_to_images[cls]
        if len(imgs) > min_count:
            imgs = random.sample(imgs, min_count)
        balanced_image_paths.extend(imgs)
        balanced_labels.extend([class_indices[cls]] * len(imgs))
    
    return balanced_image_paths, balanced_labels


def load_and_preprocess_image(path, img_height, img_width, preprocess_fn):
    """
    Carrega uma imagem a partir de um caminho, decodifica, redimensiona e normaliza.

    Parâmetros:
    - path (str): Caminho para a imagem.
    - img_height (int): Altura para redimensionamento da imagem.
    - img_width (int): Largura para redimensionamento da imagem.
    - preprocess_fn (function): Função de pré-processamento específica do modelo.

    Retorna:
    - image (tensor): Imagem pré-processada.
    """
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [img_height, img_width])
    image = preprocess_fn(image)
    return image

def preprocess_train(image, label, img_height, img_width, preprocess_fn):
    """
    Função de pré-processamento para o conjunto de treinamento.

    Parâmetros:
    - image (str): Caminho para a imagem.
    - label (int): Label correspondente.
    - img_height (int): Altura para redimensionamento da imagem.
    - img_width (int): Largura para redimensionamento da imagem.
    - preprocess_fn (function): Função de pré-processamento específica do modelo.

    Retorna:
    - image (tensor): Imagem pré-processada.
    - label (tensor): Label correspondente.
    """
    image = load_and_preprocess_image(image, img_height, img_width, preprocess_fn)
    return image, label

def preprocess_test(image, label, img_height, img_width, preprocess_fn):
    """
    Função de pré-processamento para o conjunto de teste.

    Parâmetros:
    - image (str): Caminho para a imagem.
    - label (int): Label correspondente.
    - img_height (int): Altura para redimensionamento da imagem.
    - img_width (int): Largura para redimensionamento da imagem.
    - preprocess_fn (function): Função de pré-processamento específica do modelo.

    Retorna:
    - image (tensor): Imagem pré-processada.
    - label (tensor): Label correspondente.
    """
    image = load_and_preprocess_image(image, img_height, img_width, preprocess_fn)
    return image, label

def create_datasets(train_dir, test_dir, classes, preprocess_fn, batch_size=32, img_height=256, img_width=256, shuffle_buffer=1000):
    """
    Cria os datasets de treinamento e teste a partir dos diretórios especificados.

    Parâmetros:
    - train_dir (str): Diretório das imagens de treinamento.
    - test_dir (str): Diretório das imagens de teste.
    - classes (list): Lista de nomes das classes.
    - preprocess_fn (function): Função de pré-processamento específica do modelo.
    - batch_size (int): Tamanho do batch para treinamento.
    - img_height (int): Altura para redimensionamento das imagens.
    - img_width (int): Largura para redimensionamento das imagens.
    - shuffle_buffer (int): Tamanho do buffer para embaralhamento.

    Retorna:
    - train_ds (tf.data.Dataset): Dataset de treinamento.
    - test_ds (tf.data.Dataset): Dataset de teste.
    - train_image_paths_balanced (list): Lista balanceada de caminhos das imagens de treinamento.
    - train_labels_balanced (list): Lista balanceada de labels de treinamento.
    """
    # Definir mapeamento de classes
    class_indices = {cls_name: idx for idx, cls_name in enumerate(classes)}

    # Obter caminhos e labels para treinamento
    train_image_paths, train_labels = get_image_paths_and_labels(train_dir, classes)

    # Balancear o conjunto de treinamento
    train_image_paths_balanced, train_labels_balanced = balance_dataset(train_image_paths, train_labels, classes)
    print(f"Conjunto de treinamento balanceado possui {len(train_image_paths_balanced)} imagens.")

    # Contar e imprimir o número de imagens por classe no conjunto de treinamento balanceado
    counter = Counter(train_labels_balanced)
    print("Distribuição de imagens por classe no conjunto de treinamento balanceado:")
    for cls in classes:
        cls_idx = class_indices[cls]
        count = counter.get(cls_idx, 0)
        print(f"  Classe '{cls}': {count} imagens")

    # Obter caminhos e labels para teste (sem balanceamento)
    test_image_paths, test_labels = get_image_paths_and_labels(test_dir, classes)
    print(f"Conjunto de teste possui {len(test_image_paths)} imagens.")

    # Converter listas para tensores
    train_image_paths_tensor = tf.constant(train_image_paths_balanced)
    train_labels_tensor = tf.constant(train_labels_balanced)

    test_image_paths_tensor = tf.constant(test_image_paths)
    test_labels_tensor = tf.constant(test_labels)

    # Criar o dataset de treinamento
    train_ds = tf.data.Dataset.from_tensor_slices((train_image_paths_tensor, train_labels_tensor))
    train_ds = train_ds.shuffle(buffer_size=shuffle_buffer, seed=42)
    train_ds = train_ds.map(lambda x, y: preprocess_train(x, y, img_height, img_width, preprocess_fn), num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Criar o dataset de teste
    test_ds = tf.data.Dataset.from_tensor_slices((test_image_paths_tensor, test_labels_tensor))
    test_ds = test_ds.map(lambda x, y: preprocess_test(x, y, img_height, img_width, preprocess_fn), num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_ds, test_ds, train_image_paths_balanced, train_labels_balanced
