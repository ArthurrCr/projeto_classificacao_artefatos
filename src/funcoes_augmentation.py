import tensorflow as tf
import keras_cv
import cv2
import numpy as np

# Função para carregar e pré-processar a imagem
def load_and_preprocess_image(path):
    '''
    Carrega uma imagem a partir de um caminho, decodifica, redimensiona e normaliza.

    Parâmetros:
    - path (str): Caminho para a imagem.

    Retorna:
    - image (tensor): Imagem pré-processada com dimensões (256, 256, 3) e valores normalizados entre 0 e 1.
    '''
    # Ler a imagem do arquivo
    image = tf.io.read_file(path)
    # Decodificar a imagem JPEG
    image = tf.image.decode_jpeg(image, channels=3)
    # Redimensionar a imagem para 256x256 pixels
    image = tf.image.resize(image, [256, 256])
    # Normalizar os pixels para o intervalo [0, 1]
    image = image / 255.0
    return image

def fixed_rotation(image, angle):
    '''
    Aplica uma rotação fixa à imagem no ângulo especificado usando KerasCV.

    Parâmetros:
    - image (tensor): Imagem de entrada.
    - angle (float): Ângulo para a rotação em graus (positivo para anti-horário, negativo para horário).

    Retorna:
    - image (tensor): Imagem rotacionada.
    '''
    # Converte o ângulo de graus para fração de rotação (KerasCV espera um fator de rotação entre -1 e 1)
    rotation_factor = angle / 360.0
    rotation_layer = keras_cv.layers.RandomRotation(factor=(rotation_factor, rotation_factor))
    
    # Aplicar a rotação
    rotated_image = rotation_layer(image[None, ...])[0]
    
    return rotated_image

def adjust_brightness(image, delta):
    '''
    Ajusta o brilho da imagem adicionando um deslocamento aos valores dos pixels.

    Parâmetros:
    - image (tensor): Imagem de entrada.
    - delta (float): Valor a ser adicionado ao brilho (pode ser positivo ou negativo).

    Retorna:
    - image (tensor): Imagem com brilho ajustado.
    '''
    # Ajustar o brilho da imagem
    image = tf.image.adjust_brightness(image, delta=delta)
    # Garantir que os valores dos pixels permaneçam no intervalo [0, 1]
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image

def adjust_contrast(image, contrast_factor):
    '''
    Ajusta o contraste da imagem por um fator especificado.

    Parâmetros:
    - image (tensor): Imagem de entrada.
    - contrast_factor (float): Fator para ajustar o contraste. Valores > 1 aumentam o contraste, valores < 1 diminuem.

    Retorna:
    - image (tensor): Imagem com contraste ajustado.
    '''
    image = tf.image.adjust_contrast(image, contrast_factor)
    # Clipping para garantir que os valores estejam entre 0 e 1
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image


def adjust_saturation(image, saturation_factor):
    '''
    Ajusta a saturação da imagem por um fator especificado.

    Parâmetros:
    - image (tensor): Imagem de entrada.
    - saturation_factor (float): Fator para ajustar a saturação. Valores > 1 aumentam a saturação, valores < 1 diminuem.

    Retorna:
    - image (tensor): Imagem com saturação ajustada.
    '''
    image = tf.image.adjust_saturation(image, saturation_factor)
    return image

def apply_gaussian_blur(image, kernel_size=5, sigma=1.0):
    # Converter tensor para array do NumPy
    image_np = image.numpy() if isinstance(image, tf.Tensor) else image
    blurred_image = cv2.GaussianBlur(image_np, (kernel_size, kernel_size), sigma)
    return tf.convert_to_tensor(blurred_image, dtype=tf.float32)


def alter_color(image, hue_delta=0.1):
    '''
    Altera a cor da imagem ajustando apenas a matiz.

    Parâmetros:
    - image (tensor): Imagem de entrada com dimensões (altura, largura, canais) e valores normalizados entre 0 e 1.
    - hue_delta (float): Valor a ser adicionado à matiz. Deve estar entre -0.5 e 0.5.

    Retorna:
    - altered_image (tensor): Imagem com matiz alterada.
    '''
    # Verificar se a imagem tem 3 canais (RGB)
    if image.shape[-1] != 3:
        raise ValueError(f"Esperado que a imagem tenha 3 canais (RGB), mas recebeu {image.shape[-1]} canais.")
    
    # Ajustar a matiz
    altered_image = tf.image.adjust_hue(image, hue_delta)
    
    # Garantir que os valores dos pixels permaneçam no intervalo [0, 1]
    altered_image = tf.clip_by_value(altered_image, 0.0, 1.0)
    
    return altered_image


def apply_fixed_zoom(image, zoom_factor=1.2):
    '''
    Aplica um zoom fixo à imagem, recortando e redimensionando para manter as dimensões originais.

    Parâmetros:
    - image (tensor): Imagem de entrada com dimensões (altura, largura, canais).
    - zoom_factor (float): Fator de zoom fixo. Valores > 1 aumentam o zoom (zoom in), valores < 1 reduzem (zoom out).

    Retorna:
    - zoomed_image (tensor): Imagem com zoom fixo aplicado.
    '''
    if zoom_factor <= 0:
        raise ValueError("O fator de zoom deve ser maior que 0.")

    height, width, _ = image.shape

    # Calcular a área de recorte central
    crop_height = int(height / zoom_factor)
    crop_width = int(width / zoom_factor)

    # Coordenadas para recortar a área central
    start_y = (height - crop_height) // 2
    start_x = (width - crop_width) // 2

    # Recortar a imagem
    cropped_image = tf.image.crop_to_bounding_box(
        image,
        offset_height=start_y,
        offset_width=start_x,
        target_height=crop_height,
        target_width=crop_width
    )

    # Redimensionar a imagem recortada de volta para as dimensões originais
    zoomed_image = tf.image.resize(cropped_image, [height, width])

    # Garantir que os valores dos pixels estejam no intervalo [0, 1]
    zoomed_image = tf.clip_by_value(zoomed_image, 0.0, 1.0)

    return zoomed_image

def mixup_batch(images, labels, alpha=0.5, seed=42):
    '''
    Aplica a técnica de MixUp em um batch de imagens e seus rótulos correspondentes.

    Parâmetros:
    - images (tensor): Batch de imagens de entrada com dimensões (batch_size, altura, largura, canais).
    - labels (tensor): Batch de rótulos correspondentes.
    - alpha (float): Parâmetro da distribuição Beta que controla a intensidade da mistura.
    - seed (int): Semente para garantir reprodutibilidade.

    Retorna:
    - mixed_images (tensor): Batch de imagens resultantes da combinação.
    - mixed_labels (tensor): Batch de rótulos resultantes da combinação.
    '''
    np.random.seed(seed)
    tf.random.set_seed(seed)

    batch_size = tf.shape(images)[0]

    # Gera índices aleatórios para embaralhar o batch
    indices = tf.random.shuffle(tf.range(batch_size))

    # Embaralha as imagens e os rótulos
    shuffled_images = tf.gather(images, indices)
    shuffled_labels = tf.gather(labels, indices)

    # Amostra o parâmetro lambda da distribuição Beta
    lam = np.random.beta(alpha, alpha, batch_size)
    lam = tf.convert_to_tensor(lam, dtype=tf.float32)

    # Redimensiona lambda para que possa ser usado na operação de broadcast
    lam = tf.reshape(lam, (batch_size, 1, 1, 1))

    # Cria as imagens e os rótulos misturados
    mixed_images = lam * images + (1 - lam) * shuffled_images
    mixed_labels = lam[:, 0] * labels + (1 - lam[:, 0]) * shuffled_labels

    return mixed_images, mixed_labels


def cutmix_batch(images, labels, alpha=1.0):
    '''
    Aplica a técnica de CutMix em um batch de imagens e seus rótulos correspondentes.

    Parâmetros:
    - images (tensor): Batch de imagens de entrada com dimensões (batch_size, altura, largura, canais).
    - labels (tensor): Batch de rótulos correspondentes.
    - alpha (float): Parâmetro da distribuição Beta que controla o tamanho da área recortada.

    Retorna:
    - mixed_images (tensor): Batch de imagens resultantes da combinação.
    - mixed_labels (tensor): Batch de rótulos resultantes da combinação.
    '''

    # Certifique-se de que as imagens estão em float32
    images = tf.cast(images, tf.float32)

    batch_size = tf.shape(images)[0]
    indices = tf.random.shuffle(tf.range(batch_size))

    shuffled_images = tf.gather(images, indices)
    shuffled_labels = tf.gather(labels, indices)

    # Amostrar 'lam' para cada imagem no batch
    lam = tfp.distributions.Beta(alpha, alpha).sample([batch_size])

    # Converter height e width para int32
    height = tf.shape(images)[1]
    width = tf.shape(images)[2]

    # Calcular o tamanho do corte diretamente proporcional a (1 - lam)
    cut_rat = 1. - lam
    cut_w = tf.cast(width * cut_rat, tf.int32)
    cut_h = tf.cast(height * cut_rat, tf.int32)

    # Definir o centro do recorte para cada imagem
    cx = tf.random.uniform([batch_size], 0, width, dtype=tf.int32)
    cy = tf.random.uniform([batch_size], 0, height, dtype=tf.int32)

    # Coordenadas do retângulo recortado para cada imagem
    x1 = tf.clip_by_value(cx - cut_w // 2, 0, width)
    y1 = tf.clip_by_value(cy - cut_h // 2, 0, height)
    x2 = tf.clip_by_value(cx + cut_w // 2, 0, width)
    y2 = tf.clip_by_value(cy + cut_h // 2, 0, height)

    # Criar máscaras para cada imagem no batch
    masks = []
    for i in range(batch_size):
        w = x2[i] - x1[i]
        h = y2[i] - y1[i]
        if w == 0 or h == 0:
            # Se a área de recorte é zero, crie uma máscara de zeros
            mask = tf.zeros((height, width, 1), dtype=tf.float32)
        else:
            mask = tf.ones((h, w, 1), dtype=tf.float32)
            mask = tf.image.pad_to_bounding_box(
                mask,
                y1[i],
                x1[i],
                height,
                width
            )
        masks.append(mask)
    masks = tf.stack(masks)

    # Aplicar a máscara às imagens
    mixed_images = images * (1 - masks) + shuffled_images * masks

    # Ajustar 'lam' com base na área efetiva de recorte
    area = (x2 - x1) * (y2 - y1)
    total_area = width * height
    lam_adjusted = 1 - (area / total_area)

    lam_adjusted = tf.reshape(lam_adjusted, (batch_size, 1))
    mixed_labels = lam_adjusted * labels + (1 - lam_adjusted) * shuffled_labels

    return mixed_images, mixed_labels