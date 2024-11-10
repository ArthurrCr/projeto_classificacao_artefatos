import tensorflow as tf

def load_and_preprocess_image_before_augmentation(path, img_height=256, img_width=256):
    """
    Carrega uma imagem a partir de um caminho de arquivo, decodifica, redimensiona e retorna a imagem.

    Args:
        path (tf.Tensor): Caminho para a imagem.
        img_height (int): Altura para redimensionamento da imagem.
        img_width (int): Largura para redimensionamento da imagem.

    Returns:
        tf.Tensor: Imagem pré-processada.
    """
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [img_height, img_width])
    image = tf.cast(image, tf.float32)  # Converter para float32 para compatibilidade com preprocess_input
    return image  # Não aplica preprocess_input aqui

def create_augmented_dataset(train_image_paths, train_labels, augmentation_function, batch_size=32, img_height=256, img_width=256, shuffle_buffer=1000):
    """
    Cria o dataset com Data Augmentation para treinamento.

    Args:
        train_image_paths (tf.Tensor): Tensor contendo os caminhos das imagens de treinamento.
        train_labels (tf.Tensor): Tensor contendo as labels de treinamento.
        augmentation_function (function): Função de augmentation a ser aplicada às imagens.
        batch_size (int, opcional): Tamanho do batch. Padrão é 32.
        img_height (int, opcional): Altura para redimensionamento das imagens. Padrão é 256.
        img_width (int, opcional): Largura para redimensionamento das imagens. Padrão é 256.
        shuffle_buffer (int, opcional): Tamanho do buffer para embaralhamento. Padrão é 1000.

    Returns:
        tf.data.Dataset: Dataset de treinamento com Data Augmentation.
    """
    ds = tf.data.Dataset.from_tensor_slices((train_image_paths, train_labels))
    ds = ds.map(
        lambda x, y: (load_and_preprocess_image_before_augmentation(x, img_height, img_width), y),
        num_parallel_calls=tf.data.AUTOTUNE
    ).map(
        lambda x, y: (augmentation_function(x), y),
        num_parallel_calls=tf.data.AUTOTUNE
    ).shuffle(buffer_size=shuffle_buffer, seed=42)  # Shuffle para garantir aleatoriedade
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

def create_mixup_dataset(train_image_paths, train_labels, augmentation_function,batch_size=32, img_height=256, img_width=256, shuffle_buffer=1000, alpha=0.2):
    """
    Cria o dataset de treinamento aplicando a técnica de MixUp.

    Args:
        train_image_paths (tf.Tensor): Tensor contendo os caminhos das imagens de treinamento.
        train_labels (tf.Tensor): Tensor contendo as labels de treinamento.
        batch_size (int, opcional): Tamanho do batch. Padrão é 32.
        img_height (int, opcional): Altura para redimensionamento das imagens. Padrão é 256.
        img_width (int, opcional): Largura para redimensionamento das imagens. Padrão é 256.
        shuffle_buffer (int, opcional): Tamanho do buffer para embaralhamento. Padrão é 1000.
        alpha (float, opcional): Parâmetro da distribuição Beta para MixUp. Padrão é 0.2.

    Returns:
        tf.data.Dataset: Dataset de treinamento com MixUp aplicado.
    """
    def load_and_preprocess_image(path):
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [img_height, img_width])
        image = tf.cast(image, tf.float32)
        return image

    ds = tf.data.Dataset.from_tensor_slices((train_image_paths, train_labels))
    ds = ds.map(
        lambda x, y: (load_and_preprocess_image(x), y),
        num_parallel_calls=tf.data.AUTOTUNE
    ).shuffle(buffer_size=shuffle_buffer, seed=42)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Aplicar MixUp após o batch
    ds = ds.map(lambda images, labels: augmentation_function(images, labels, alpha=alpha), num_parallel_calls=tf.data.AUTOTUNE)

    return ds
