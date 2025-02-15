import torch
import torch.nn as nn
import timm

def create_model_vgg16(num_classes: int, neuron: int, dropout: float) -> nn.Sequential:
    """
    Cria um modelo VGG16 customizado usando timm.
    
    - Congela todas as camadas do modelo pré-treinado.
    - Remove a última camada de classificação.
    - Determina dinamicamente o número de features de saída.
    - Adiciona um flatten e uma cabeça personalizada (sem Softmax).
    
    Parâmetros:
      num_classes: número de classes de saída.
      neuron: número de neurônios na camada intermediária da nova cabeça.
      dropout: taxa de dropout.
      
    Retorna:
      Um nn.Sequential contendo o modelo base seguido da nova cabeça.
    """
    base_model = timm.create_model('vgg16', pretrained=True)

    # Congelar todas as camadas
    for param in base_model.parameters():
        param.requires_grad = False

    # Remover a última camada de classificação
    base_model.reset_classifier(0, '')  # O modelo agora retorna features em (N, C, H, W)

    # Testar com um dummy input para determinar o número de features de saída
    dummy_input = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        dummy_output = base_model(dummy_input)  # Saída: (1, C, H, W)
    c, h, w = dummy_output.shape[1], dummy_output.shape[2], dummy_output.shape[3]
    in_features = c * h * w

    # Cabeça personalizada: flatten + camada(s) fully-connected
    head = nn.Sequential(
        nn.BatchNorm1d(in_features),
        nn.Dropout(dropout),
        nn.Linear(in_features, neuron),
        nn.ReLU(),
        nn.BatchNorm1d(neuron),
        nn.Dropout(dropout),
        nn.Linear(neuron, num_classes)
    )

    # Combina o backbone, um flatten e a nova cabeça
    model = nn.Sequential(
        base_model,
        nn.Flatten(1),  # Converte de (N, C, H, W) para (N, C*H*W)
        head
    )
    return model


def create_model_vgg19(num_classes: int, neuron: int, dropout: float) -> nn.Sequential:
    """
    Cria um modelo VGG19 customizado usando timm.
    
    - Congela todas as camadas do modelo pré-treinado.
    - Remove a última camada de classificação.
    - Determina dinamicamente o número de features de saída.
    - Adiciona um flatten e uma cabeça personalizada (sem Softmax).
    
    Parâmetros:
      num_classes: número de classes de saída.
      neuron: número de neurônios na camada intermediária da nova cabeça.
      dropout: taxa de dropout.
      
    Retorna:
      Um nn.Sequential contendo o modelo base seguido da nova cabeça.
    """
    base_model = timm.create_model('vgg19', pretrained=True)

    # Congelar todas as camadas
    for param in base_model.parameters():
        param.requires_grad = False

    # Remover a última camada de classificação original
    base_model.reset_classifier(0, '')  # O modelo agora retorna features em (N, C, H, W)

    # Calcular dinamicamente o número de features de saída do backbone
    dummy_input = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        dummy_output = base_model(dummy_input)  # Saída: (1, C, H, W)
    c, h, w = dummy_output.shape[1], dummy_output.shape[2], dummy_output.shape[3]
    in_features = c * h * w

    # Cabeça personalizada (sem flatten extra, pois já usamos um flatten fora)
    head = nn.Sequential(
        nn.BatchNorm1d(in_features),
        nn.Dropout(dropout),
        nn.Linear(in_features, neuron),
        nn.ReLU(),
        nn.BatchNorm1d(neuron),
        nn.Dropout(dropout),
        nn.Linear(neuron, num_classes)
    )

    # Combina o backbone, um flatten e a nova cabeça
    model = nn.Sequential(
        base_model,
        nn.Flatten(1),  # Converte de (N, C, H, W) para (N, C*H*W)
        head
    )
    return model
