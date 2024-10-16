import os
import tensorflow as tf
import numpy as np
import tensorflow_addons as tfa

# Função para carregar e pré-processar a imagem
def load_and_preprocess_image(path):
    # Ler a imagem
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    # Redimensionar a imagem
    image = tf.image.resize(image, [256, 256])
    # Normalizar os pixels para [0, 1]
    image = image / 255.0
    return image

def random_rotation(image, max_angle):
    # Converter o ângulo para radianos
    angle = tf.random.uniform([], -max_angle, max_angle, dtype=tf.float32) * np.pi / 180
    image = tfa.image.rotate(image, angles=angle, fill_mode='nearest')
    return image

def adjust_brightness(image, delta):
    image = tf.image.adjust_brightness(image, delta=delta)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image

def flip_horizontal(image):
    image = tf.image.flip_left_right(image)
    return image

def random_zoom(image, zoom_range=(0.8, 1.2)):
    # Random zoom factor
    zoom = tf.random.uniform([], zoom_range[0], zoom_range[1])
    # Get image dimensions
    height, width = tf.shape(image)[0], tf.shape(image)[1]
    # Compute new dimensions
    new_height = tf.cast(zoom * tf.cast(height, tf.float32), tf.int32)
    new_width = tf.cast(zoom * tf.cast(width, tf.float32), tf.int32)
    # Resize image
    image = tf.image.resize(image, [new_height, new_width])
    # Crop or pad to original size
    image = tf.image.resize_with_crop_or_pad(image, height, width)
    return image

def random_shear(image, shear_level):
    # shear_level é o ângulo de cisalhamento em radianos
    shear = tf.random.uniform([], -shear_level, shear_level)

    # Criar a matriz de transformação
    shear_matrix = [1.0, -tf.math.sin(shear), 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0]

    # Aplicar a transformação
    image = tfa.image.transform(image, shear_matrix, interpolation='BILINEAR', fill_mode='nearest')
    return image

