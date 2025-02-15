import os
import pickle
import numpy as np
import statistics
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold

from src.preprocess import get_preprocess_fn
from src.dataset import BalancedImageDataset
from src.funcoes_augmentation import get_augmentation_fn
from src.modelos.efficientnet import create_model_v2l 
from src.trainer import train_model

# ---------------------------
# Gerenciamento de Estado
# ---------------------------
def load_experiment_state(state_path, param_values, param_name):
    """
    Carrega o estado salvo ou inicializa um novo estado.
    """
    if os.path.exists(state_path):
        with open(state_path, 'rb') as f:
            state = pickle.load(f)
        print("Estado carregado com sucesso.")
    else:
        state = {
            'all_results': {val: [] for val in param_values},
            f'current_{param_name}': None,
            'current_rep': None,
            'current_fold': None
        }
    return state

def save_experiment_state(state, state_path):
    """
    Salva o estado atual do experimento.
    """
    with open(state_path, 'wb') as f:
        pickle.dump(state, f)

def run_augmentation_experiments_from_loader(
    train_loader,                 # DataLoader já criado para o conjunto de treino
    classes: list,
    batch_size: int,
    img_height: int,
    img_width: int,
    mean: list,
    std: list,
    aug_param_name: str,          # Ex: "rotation_degrees", "brightness", etc.
    aug_param_values: list,       # Lista de valores para o parâmetro a variar
    fixed_aug_params: dict = None,  # Parâmetros fixos para os demais aumentos
    k: int = 5,
    num_repeticoes: int = 5,
    state_path: str = '/content/drive/MyDrive/projeto_classificacao_artefatos/estado_treinamento_aug.pkl'
):
    if fixed_aug_params is None:
        fixed_aug_params = {}

    # Carrega ou inicializa o estado do experimento
    state = load_experiment_state(state_path, aug_param_values, aug_param_name)

    # Obtém o dataset a partir do train_loader.
    # Se o dataset for um Subset, obtenha o dataset original.
    dataset = train_loader.dataset
    if hasattr(dataset, "dataset"):
        dataset = dataset.dataset

    # Agora, assumindo que o dataset possui os atributos image_paths e labels:
    train_labels_balanced = dataset.labels

    # Para cada valor do parâmetro a ser variado, realiza as repetições e os folds
    for aug_value in aug_param_values:
        if state[f'current_{aug_param_name}'] is not None and aug_value < state[f'current_{aug_param_name}']:
            continue
        print(f"\nTreinando com {aug_param_name} = {aug_value}\n")
        if aug_value not in state['all_results']:
            state['all_results'][aug_value] = []
        inicio_rep = state['current_rep'] if (state[f'current_{aug_param_name}'] == aug_value and state['current_rep'] is not None) else 0

        for rep in range(inicio_rep, num_repeticoes):
            print(f"  Repetição {rep+1}/{num_repeticoes} para {aug_param_name} = {aug_value}")
            fold_results = []
            # Gera índices para o dataset
            indices = np.arange(len(dataset))
            skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42 + rep)
            inicio_fold = state['current_fold'] if (state[f'current_{aug_param_name}'] == aug_value and 
                                                    state['current_rep'] == rep and 
                                                    state['current_fold'] is not None) else 0
            for fold, (train_idx, val_idx) in enumerate(skf.split(indices, train_labels_balanced)):
                if fold < inicio_fold:
                    continue
                print(f"    Fold {fold+1}/{k}")

                # Cria subsets de treino e validação a partir do dataset original
                # Aplicando transformações diferentes para treino (com augmentação) e validação (pré-processamento padrão)
                aug_params = fixed_aug_params.copy()
                aug_params[aug_param_name] = aug_value
                augmentation_fn_train = get_augmentation_fn(
                    img_height=img_height,
                    img_width=img_width,
                    mean=mean,
                    std=std,
                    **aug_params
                )
                preprocess_fn = get_preprocess_fn(img_height=img_height, img_width=img_width)

                from torch.utils.data import Subset
                train_dataset = BalancedImageDataset(
                    image_paths=[dataset.image_paths[i] for i in train_idx],
                    labels=[dataset.labels[i] for i in train_idx],
                    transform=augmentation_fn_train
                )
                val_dataset = BalancedImageDataset(
                    image_paths=[dataset.image_paths[i] for i in val_idx],
                    labels=[dataset.labels[i] for i in val_idx],
                    transform=preprocess_fn
                )

                train_loader_fold = torch.utils.data.DataLoader(
                    train_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=4,
                    pin_memory=True
                )
                val_loader_fold = torch.utils.data.DataLoader(
                    val_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=4,
                    pin_memory=True
                )

                model, train_losses, val_losses, train_accuracies, val_accuracies = train_model(
                    model_fn=create_model_v2l,
                    train_loader=train_loader_fold,
                    val_loader=val_loader_fold,
                    classes=classes,
                    neuron=128,
                    dropout=0.8,
                    epochs=10,
                    lr=1e-4,
                    save_path=f'model_{aug_param_name}_{aug_value}_rep_{rep+1}_fold_{fold+1}.pt'
                )
                fold_results.append((val_losses[-1], val_accuracies[-1]))

                state[f'current_{aug_param_name}'] = aug_value
                state['current_rep'] = rep
                state['current_fold'] = fold
                save_experiment_state(state, state_path)

            if fold_results:
                mean_val_loss = np.mean([r[0] for r in fold_results])
                mean_val_acc = np.mean([r[1] for r in fold_results])
                if aug_value not in state['all_results']:
                    state['all_results'][aug_value] = []
                if len(state['all_results'][aug_value]) > rep:
                    state['all_results'][aug_value][rep] = (mean_val_loss, mean_val_acc)
                else:
                    state['all_results'][aug_value].append((mean_val_loss, mean_val_acc))
                state['current_fold'] = None
                save_experiment_state(state, state_path)
        # Fim do loop para um valor de augmentação

    # Cálculo dos resultados médios e desvios padrão
    avg_results = {}
    for param_val, results in state['all_results'].items():
        if results:
            val_losses = [r[0] for r in results]
            val_accs = [r[1] for r in results]
            avg_results[param_val] = (
                statistics.mean(val_losses),
                statistics.mean(val_accs),
                statistics.stdev(val_losses) if len(val_losses) > 1 else 0.0,
                statistics.stdev(val_accs) if len(val_accs) > 1 else 0.0
            )
    print("\nResultados Médios após todas as repetições:")
    for param_val, (mean_loss, mean_acc, std_loss, std_acc) in avg_results.items():
        print(f"{aug_param_name}: {param_val} | Val Loss: {mean_loss:.4f} ± {std_loss:.4f} | Val Acc: {mean_acc:.4f} ± {std_acc:.4f}")
    best_param = max(avg_results, key=lambda p: avg_results[p][1])
    print(f"\nMelhor {aug_param_name} encontrado: {best_param} com acurácia média de validação {avg_results[best_param][1]:.4f}")

    param_values_list = list(avg_results.keys())
    mean_accs = [avg_results[p][1] for p in param_values_list]
    std_accs = [avg_results[p][3] for p in param_values_list]
    plt.figure(figsize=(10, 6))
    plt.errorbar(param_values_list, mean_accs, yerr=std_accs, fmt='-o', capsize=5, ecolor='red', color='blue')
    plt.title(f"Acurácia média de validação vs. {aug_param_name}")
    plt.xlabel(aug_param_name)
    plt.ylabel("Acurácia média de validação")
    plt.grid(True)
    plt.show()

    return avg_results
