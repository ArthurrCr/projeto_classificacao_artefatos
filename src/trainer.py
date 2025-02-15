import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import copy

def train_model(
    model_fn,
    train_loader,
    val_loader,
    classes,
    neuron,
    dropout,
    device="cuda",
    epochs=10,
    lr=1e-3,
    mixup_fn=None,  # função de Mixup
    save_path='modelo_treinado.pt',
    eval_train_accuracy=True  # se True, avalia a acurácia no train_loader após cada época
):
    """
    Treina um modelo PyTorch usando os dataloaders fornecidos, com suporte para Mixup e AMP para aceleração.
    
    Parâmetros:
      model_fn: função que cria e retorna o modelo (ex: create_model_b0).
      train_loader: DataLoader para o conjunto de treino.
      val_loader: DataLoader para o conjunto de validação.
      classes: lista com os nomes das classes.
      neuron: número de neurônios na camada intermediária.
      dropout: taxa de dropout.
      device: 'cuda' ou 'cpu'. Se None, detecta automaticamente.
      epochs: número de épocas.
      lr: taxa de aprendizado.
      mixup_fn: função de Mixup (se None, não aplica Mixup).
      save_path: caminho para salvar o modelo.
      eval_train_accuracy: se True, avalia a acurácia no train_loader após cada época.
      
    Retorna:
      model, train_losses, val_losses, train_accuracies, val_accuracies
    """
    # Converte device para torch.device se necessário
    if isinstance(device, str):
        device = torch.device(device)
    elif device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    num_classes = len(classes)
    model = model_fn(num_classes, neuron, dropout).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Define a função de perda
    if mixup_fn is not None:
        def criterion(outputs, targets):
            return (-targets * F.log_softmax(outputs, dim=1)).sum(dim=1).mean()
    else:
        criterion = nn.CrossEntropyLoss()
    
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    best_val_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    best_epoch = 0

    # Inicializa o GradScaler usando a nova API
    scaler = torch.amp.GradScaler(device=device) if device.type == 'cuda' else None

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        train_samples = 0

        for images, labels in tqdm(train_loader, desc=f"Treinando época {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            
            # Usa autocast com a nova API; se não estiver em GPU, desativa
            with torch.amp.autocast(device_type='cuda' if device.type=='cuda' else 'cpu'):
                if mixup_fn is not None:
                    images, mixed_labels = mixup_fn(images, labels)
                    outputs = model(images)
                    loss = criterion(outputs, mixed_labels)
                else:
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            train_samples += batch_size
        
        train_loss = running_loss / train_samples

        # Avaliação no conjunto de validação 
        model.eval()
        running_val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                with torch.amp.autocast(device_type='cuda' if device.type=='cuda' else 'cpu'):
                    outputs = model(images)
                    loss = F.cross_entropy(outputs, labels)
                running_val_loss += loss.item() * images.size(0)
                preds = torch.argmax(outputs, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss = running_val_loss / val_total
        val_acc = val_correct / val_total
        
        # Calcula acurácia no conjunto de treino 
        if eval_train_accuracy:
            model.eval()
            train_correct = 0
            train_total = 0
            with torch.no_grad():
                for images, labels in train_loader:
                    images, labels = images.to(device), labels.to(device)
                    with torch.amp.autocast(device_type='cuda' if device.type=='cuda' else 'cpu'):
                        outputs = model(images)
                    preds = torch.argmax(outputs, dim=1)
                    train_correct += (preds == labels).sum().item()
                    train_total += labels.size(0)
            train_acc = train_correct / train_total
        else:
            train_acc = None

        if train_acc is not None:
            print(f"Época [{epoch+1}/{epochs}] | Loss Treino: {train_loss:.4f}, Acc Treino: {train_acc:.4f}, Loss Val: {val_loss:.4f}, Acc Val: {val_acc:.4f}")
        else:
            print(f"Época [{epoch+1}/{epochs}] | Loss Treino: {train_loss:.4f} | Loss Val: {val_loss:.4f} | Acc Val: {val_acc:.4f}")
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1

    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), save_path)
    print(f"Modelo com melhor Acc Val ({best_val_acc:.4f}) encontrado na época {best_epoch}. Salvo em: {save_path}")

    return model, train_losses, val_losses, train_accuracies, val_accuracies
