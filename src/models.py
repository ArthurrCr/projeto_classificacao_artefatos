from tensorflow.keras.applications import EfficientNetB0, EfficientNetB1
from tensorflow.keras import layers, models
import tensorflow as tf
from tensorflow.keras.applications import VGG16, VGG19
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras import backend as K


def create_model_b0(num_classes, img_height, img_width):
    '''
    Cria e retorna um modelo de classificação baseado em EfficientNetB0 com possíveis melhorias.

    Parâmetros:
    - num_classes (int): Número de classes para a classificação.
    - img_height (int): Altura das imagens de entrada.
    - img_width (int): Largura das imagens de entrada.

    Retorna:
    - model (tf.keras.Model): Modelo pronto para treinamento.
    '''

    # Limpar a sessão e definir a semente
    K.clear_session()
    tf.random.set_seed(42)

    # Especificar o inicializador com uma semente
    initializer = GlorotUniform(seed=42)

    # Carregar o modelo base EfficientNetB0 pré-treinado no ImageNet
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(img_height, img_width, 3)
    )

    # Congelar todas as camadas do modelo base inicialmente
    base_model.trainable = False

    # Definir o ponto de fine-tuning (descongelar as últimas 10 camadas)
    fine_tune_at = len(base_model.layers) - 10

    # Congelar todas as camadas antes do ponto de fine-tuning
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    # Descongelar as camadas a partir do ponto de fine-tuning
    for layer in base_model.layers[fine_tune_at:]:
        layer.trainable = True

    # Construir a arquitetura do modelo
    inputs = layers.Input(shape=(img_height, img_width, 3))
    x = base_model(inputs)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5, seed=42)(x)  # Definir a semente na camada Dropout
    x = layers.Dense(
        64,
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(0.0001),
        kernel_initializer=initializer  # Usar o inicializador com semente
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3, seed=42)(x)  # Definir a semente na camada Dropout
    outputs = layers.Dense(
        num_classes,
        activation='softmax',
        kernel_initializer=initializer  # Usar o inicializador com semente
    )(x)
    model = models.Model(inputs, outputs)

    return model


def create_model_b1(num_classes, img_height, img_width):
    '''
    Cria e retorna um modelo de classificação baseado em EfficientNetB1 com possíveis melhorias.

    Parâmetros:
    - num_classes (int): Número de classes para a classificação.
    - img_height (int): Altura das imagens de entrada.
    - img_width (int): Largura das imagens de entrada.

    Retorna:
    - model (tf.keras.Model): Modelo pronto para treinamento.
    '''
    # Limpar a sessão Keras e definir a semente
    K.clear_session()
    tf.random.set_seed(42)

    # Especificar o inicializador com uma semente
    initializer = GlorotUniform(seed=42)

    # Carregar o modelo base EfficientNetB1 pré-treinado no ImageNet
    base_model = EfficientNetB1(
        weights='imagenet',
        include_top=False,
        input_shape=(img_height, img_width, 3)
    )

    # Congelar todas as camadas do modelo base inicialmente
    base_model.trainable = False

    # Definir o ponto de fine-tuning (descongelar as últimas 10 camadas)
    fine_tune_at = len(base_model.layers) - 10

    # Congelar todas as camadas antes do ponto de fine-tuning
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    # Descongelar as camadas a partir do ponto de fine-tuning
    for layer in base_model.layers[fine_tune_at:]:
        layer.trainable = True

    # Construir a arquitetura do modelo
    inputs = layers.Input(shape=(img_height, img_width, 3))
    x = base_model(inputs)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5, seed=42)(x)  # Adicionar dropout com semente
    x = layers.Dense(
        64,
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(0.0001),
        kernel_initializer=initializer  # Usar inicializador com semente
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3, seed=42)(x)  # Adicionar dropout com semente
    outputs = layers.Dense(
        num_classes,
        activation='softmax',
        kernel_initializer=initializer  # Usar inicializador com semente
    )(x)
    model = models.Model(inputs, outputs)

    return model

def create_model_vgg16(num_classes, img_height, img_width):
    '''
    Cria e retorna um modelo de classificação baseado em VGG16 com possíveis melhorias.

    Parâmetros:
    - num_classes (int): Número de classes para a classificação.
    - img_height (int): Altura das imagens de entrada.
    - img_width (int): Largura das imagens de entrada.

    Retorna:
    - model (tf.keras.Model): Modelo pronto para treinamento.
    '''
    # Limpar a sessão Keras e definir a semente
    K.clear_session()
    tf.random.set_seed(42)

    # Especificar o inicializador com uma semente
    initializer = GlorotUniform(seed=42)

    # Carregar o modelo base VGG16 pré-treinado no ImageNet
    base_model = VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=(img_height, img_width, 3)
    )

    # Congelar todas as camadas do modelo base inicialmente
    base_model.trainable = False

    # Definir o ponto de fine-tuning 
    fine_tune_at = len(base_model.layers) - 1

    # Congelar todas as camadas antes do ponto de fine-tuning
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    # Descongelar as camadas a partir do ponto de fine-tuning
    for layer in base_model.layers[fine_tune_at:]:
        layer.trainable = True

    # Construir a arquitetura do modelo
    inputs = layers.Input(shape=(img_height, img_width, 3))
    x = base_model(inputs)
    x = layers.Flatten()(x)  # VGG16 geralmente usa Flatten após as camadas convolucionais
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5, seed=42)(x)  # Adicionar dropout com semente
    x = layers.Dense(
        8,
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(0.0001),
        kernel_initializer=initializer  # Usar inicializador com semente
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5, seed=42)(x)  # Adicionar dropout com semente
    outputs = layers.Dense(
        num_classes,
        activation='softmax',
        kernel_initializer=initializer  # Usar inicializador com semente
    )(x)
    model = models.Model(inputs, outputs)

    return model


def create_model_vgg19(num_classes, img_height, img_width):
    '''
    Cria e retorna um modelo de classificação baseado em VGG19 com possíveis melhorias.

    Parâmetros:
    - num_classes (int): Número de classes para a classificação.
    - img_height (int): Altura das imagens de entrada.
    - img_width (int): Largura das imagens de entrada.

    Retorna:
    - model (tf.keras.Model): Modelo pronto para treinamento.
    '''
    # Limpar a sessão Keras e definir a semente
    K.clear_session()
    tf.random.set_seed(42)

    # Especificar o inicializador com uma semente
    initializer = GlorotUniform(seed=42)

    # Carregar o modelo base VGG19 pré-treinado no ImageNet
    base_model = VGG19(
        weights='imagenet',
        include_top=False,
        input_shape=(img_height, img_width, 3)
    )

    # Congelar todas as camadas do modelo base inicialmente
    base_model.trainable = False

    # Definir o ponto de fine-tuning 
    fine_tune_at = len(base_model.layers) - 1

    # Congelar todas as camadas antes do ponto de fine-tuning
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    # Descongelar as camadas a partir do ponto de fine-tuning
    for layer in base_model.layers[fine_tune_at:]:
        layer.trainable = True

    # Construir a arquitetura do modelo
    inputs = layers.Input(shape=(img_height, img_width, 3))
    x = base_model(inputs)
    x = layers.Flatten()(x)  # VGG19 geralmente usa Flatten após as camadas convolucionais
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5, seed=42)(x)  # Adicionar dropout com semente
    x = layers.Dense(
        8,
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(0.01),
        kernel_initializer=initializer  # Usar inicializador com semente
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5, seed=42)(x)  # Adicionar dropout com semente
    outputs = layers.Dense(
        num_classes,
        activation='softmax',
        kernel_initializer=initializer  # Usar inicializador com semente
    )(x)
    model = models.Model(inputs, outputs)

    return model


def create_model_mobilenetv2(num_classes, img_height, img_width):
    '''
    Cria e retorna um modelo de classificação baseado em MobileNetV2 com possíveis melhorias.

    Parâmetros:
    - num_classes (int): Número de classes para a classificação.
    - img_height (int): Altura das imagens de entrada.
    - img_width (int): Largura das imagens de entrada.

    Retorna:
    - model (tf.keras.Model): Modelo pronto para treinamento.
    '''
    # Limpar a sessão Keras e definir a semente
    K.clear_session()
    tf.random.set_seed(42)

    # Especificar o inicializador com uma semente
    initializer = GlorotUniform(seed=42)

    # Carregar o modelo base MobileNetV2 pré-treinado no ImageNet
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(img_height, img_width, 3)
    )

    # Congelar todas as camadas do modelo base inicialmente
    base_model.trainable = False

    # Definir o ponto de fine-tuning (descongelar as últimas 10 camadas)
    fine_tune_at = len(base_model.layers) - 10

    # Congelar todas as camadas antes do ponto de fine-tuning
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    # Descongelar as camadas a partir do ponto de fine-tuning
    for layer in base_model.layers[fine_tune_at:]:
        layer.trainable = True

    # Construir a arquitetura do modelo
    inputs = layers.Input(shape=(img_height, img_width, 3))
    x = base_model(inputs)
    x = layers.GlobalAveragePooling2D()(x)  # MobileNetV2 geralmente usa GlobalAveragePooling2D
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5, seed=42)(x)  # Adicionar dropout com semente
    x = layers.Dense(
        128,
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(0.0001),
        kernel_initializer=initializer  # Usar inicializador com semente
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3, seed=42)(x)  # Adicionar dropout com semente
    outputs = layers.Dense(
        num_classes,
        activation='softmax',
        kernel_initializer=initializer  # Usar inicializador com semente
    )(x)
    model = models.Model(inputs, outputs)

    return model