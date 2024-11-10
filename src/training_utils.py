import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model
import numpy as np
import json
import os
import seaborn as sns
import pandas as pd
import tensorflow as tf

def plot_histories(histories, metric='accuracy'):
    """
    Plota as métricas de treinamento para diferentes históricos.

    Args:
        histories (dict): Dicionário com os históricos de treinamento.
        metric (str, opcional): Métrica a ser plotada. Padrão é 'accuracy'.
    """
    plt.figure(figsize=(10, 6))

    for name, history in histories.items():
        epochs = range(1, len(history[metric]) + 1)
        plt.plot(epochs, history[metric], label=f'{name} {metric}')
        plt.plot(epochs, history[f'val_{metric}'], linestyle='--', label=f'{name} val_{metric}')

    plt.title(f'Training and Validation {metric.capitalize()}')
    plt.xlabel('Epochs')
    plt.ylabel(metric.capitalize())

    # Definindo o eixo x para mostrar apenas inteiros
    plt.xticks(ticks=epochs)

    plt.legend()
    plt.show()

def train_and_evaluate_model(
    create_model_func,
    train_dataset,
    validation_dataset,
    num_classes,
    epochs,
    history_path,  # caminho do arquivo de histórico
    model_checkpoint_path='best_model.keras',
    patience=5,
    restore_best_weights=True,
    img_height=256,
    img_width=256,
    loss='sparse_categorical_crossentropy'
):
    """
    Cria, compila, treina e avalia um modelo Keras, salvando o histórico em um arquivo JSON.

    Args:
        create_model_func (function): Função para criar o modelo.
        train_dataset (tf.data.Dataset): Conjunto de dados de treinamento.
        validation_dataset (tf.data.Dataset): Conjunto de dados de validação.
        num_classes (int): Número de classes para a camada de saída.
        epochs (int): Número de épocas para treinamento.
        history_path (str): Caminho para salvar o histórico de treinamento.
        model_checkpoint_path (str, opcional): Caminho para salvar o melhor modelo. Padrão é 'best_model.keras'.
        patience (int, opcional): Número de épocas sem melhoria antes de parar o treinamento. Padrão é 5.
        restore_best_weights (bool, opcional): Se True, restaura os pesos do melhor modelo após o treinamento. Padrão é True.
        img_height (int, opcional): Altura das imagens de entrada. Padrão é 256.
        img_width (int, opcional): Largura das imagens de entrada. Padrão é 256.

    Returns:
        model (tf.keras.Model): Modelo treinado.
        history (tf.keras.callbacks.History): Histórico de treinamento.
    """
    # Recriar o modelo usando a função fornecida
    model = create_model_func(num_classes, img_height, img_width)

    # Compilar o modelo com a taxa de aprendizado especificada
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy']
    )

    # Definir callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=restore_best_weights
    )
    model_checkpoint = ModelCheckpoint(
        filepath=model_checkpoint_path,
        save_best_only=True,
        monitor='val_loss'
    )

    # Treinar o modelo
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        callbacks=[early_stopping, model_checkpoint]
    )

    # Salvar o histórico de treinamento em um arquivo JSON
    with open(history_path, 'w') as f:
        json.dump(history.history, f)

    return model, history

def load_history(history_path):
    """
    Carrega o histórico de treinamento a partir de um arquivo JSON.

    Args:
        history_path (str): Caminho para o arquivo de histórico.

    Returns:
        dict: Dicionário contendo as métricas de treinamento.
    """
    with open(history_path, 'r') as f:
        history = json.load(f)
    return history

def evaluate_and_plot_metrics(model_checkpoint_path, validation_dataset, class_names, plot=False):
    """
    Avalia o modelo no conjunto de validação e plota todas as métricas fornecidas por classification_report
    além da matriz de confusão, se plot=True.

    Args:
        model_checkpoint_path (str): Caminho para o modelo salvo.
        validation_dataset (tf.data.Dataset): Conjunto de dados de validação.
        class_names (list): Lista com os nomes das classes.
        plot (bool): Se True, plota os gráficos. Se False, apenas imprime os resultados.

    Returns:
        float: A acurácia do modelo no conjunto de validação.
    """
    # Carregar o melhor modelo salvo
    model = load_model(model_checkpoint_path)

    # Avaliar o modelo no conjunto de validação
    loss, accuracy = model.evaluate(validation_dataset, verbose=0)

    # Gerar previsões
    y_true = []
    y_pred = []
    for batch in validation_dataset:
        images, labels = batch
        predictions = model.predict(images)
        predicted_classes = np.argmax(predictions, axis=1)
        y_true.extend(labels.numpy())
        y_pred.extend(predicted_classes)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Gerar relatório de classificação
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    report_str = classification_report(y_true, y_pred, target_names=class_names)
    print("Relatório de Classificação:\n", report_str)

    if plot:
        # Plotar a matriz de confusão
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names,
                    yticklabels=class_names)
        plt.xlabel('Previsto')
        plt.ylabel('Verdadeiro')
        plt.title('Matriz de Confusão')
        plt.tight_layout()
        plt.show()

    # Retornar a acurácia
    return accuracy

