import tensorflow as tf
import keras_cv

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
