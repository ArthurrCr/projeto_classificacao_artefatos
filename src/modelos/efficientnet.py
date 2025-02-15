import torch.nn as nn
import timm
import torchvision.models as models

def create_head(in_features: int, neuron: int, num_classes: int, dropout: float) -> nn.Sequential:
    """
    Cria a nova cabeça para a rede, composta por:
      - BatchNorm
      - Dropout
      - Linear para reduzir para 'neuron' unidades
      - ReLU
      - BatchNorm
      - Dropout
      - Linear para o número final de classes
    """
    head = nn.Sequential(
        nn.BatchNorm1d(in_features),
        nn.Dropout(dropout),
        nn.Linear(in_features, neuron),
        nn.ReLU(),
        nn.BatchNorm1d(neuron),
        nn.Dropout(dropout),
        nn.Linear(neuron, num_classes)
    )
    return head

def create_model_b0(num_classes: int, neuron: int, dropout: float) -> nn.Sequential:
    """
    Cria o modelo EfficientNet-B0 usando a biblioteca timm.
    - Congela todas as camadas, descongelando os últimos 2 blocos.
    - Remove o classificador original e adiciona uma nova cabeça.
    """
    base_model = timm.create_model('efficientnet_b0', pretrained=True)
    
    # Congelar todas as camadas
    for param in base_model.parameters():
        param.requires_grad = False

    # Descongelar os últimos 2 blocos
    for block in base_model.blocks[-2:]:
        for param in block.parameters():
            param.requires_grad = True

    # Remover o classificador original
    base_model.classifier = nn.Identity()
    
    in_features = base_model.num_features  # Obtém o número de features de saída
    head = create_head(in_features, neuron, num_classes, dropout)
    
    model = nn.Sequential(
        base_model,
        head
    )
    return model

def create_model_b1(num_classes: int, neuron: int, dropout: float) -> nn.Sequential:
    """
    Cria o modelo EfficientNet-B1 usando a biblioteca timm.
    - Congela todas as camadas, descongelando os parâmetros do último bloco (ou penúltimo, conforme estratégia).
    - Remove o classificador original e adiciona uma nova cabeça.
    """
    base_model = timm.create_model('efficientnet_b1', pretrained=True)
    
    # Congelar todas as camadas
    for param in base_model.parameters():
        param.requires_grad = False

    # Descongelar os parâmetros do penúltimo bloco
    for param in base_model.blocks[-2].parameters():
        param.requires_grad = True

    base_model.classifier = nn.Identity()
    in_features = base_model.num_features
    head = create_head(in_features, neuron, num_classes, dropout)
    
    model = nn.Sequential(
        base_model,
        head
    )
    return model

def create_model_b7(num_classes: int, neuron: int, dropout: float) -> nn.Sequential:
    """
    Cria o modelo EfficientNet-B7 usando a biblioteca torchvision.
    - Congela todas as camadas e descongela os últimos 2 blocos dos features.
    - Obtém o número de features a partir do classificador original e, em seguida, o substitui por uma nova cabeça.
    """
    base_model = models.efficientnet_b7(weights=models.EfficientNet_B7_Weights.DEFAULT)
    
    # Congelar todas as camadas
    for param in base_model.parameters():
        param.requires_grad = False

    # Descongelar os últimos 2 blocos dos features
    for layer in list(base_model.features[-2:]):
        for param in layer.parameters():
            param.requires_grad = True

    # Obter o número de features a partir do classificador original
    if isinstance(base_model.classifier, nn.Sequential):
        in_features = base_model.classifier[-1].in_features
    else:
        in_features = base_model.classifier.in_features

    # Remover o classificador original
    base_model.classifier = nn.Identity()

    head = create_head(in_features, neuron, num_classes, dropout)
    
    model = nn.Sequential(
        base_model,
        head
    )
    return model

def create_model_v2s(num_classes: int, neuron: int, dropout: float) -> nn.Sequential:
    """
    Cria o modelo EfficientNetV2-S usando a biblioteca torchvision.
    - Congela todas as camadas e descongela os últimos 2 blocos dos features.
    - Obtém o número de features a partir do classificador original e, em seguida, o substitui por uma nova cabeça.
    """
    base_model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
    
    # Congelar todas as camadas
    for param in base_model.parameters():
        param.requires_grad = False

    # Descongelar os últimos 2 blocos dos features
    for layer in list(base_model.features[-2:]):
        for param in layer.parameters():
            param.requires_grad = True

    # Obter o número de features antes de remover o classificador
    if hasattr(base_model, 'num_features'):
        in_features = base_model.num_features
    elif isinstance(base_model.classifier, nn.Sequential):
        in_features = base_model.classifier[-1].in_features
    else:
        in_features = base_model.classifier.in_features

    # Remover o classificador original
    base_model.classifier = nn.Identity()

    head = create_head(in_features, neuron, num_classes, dropout)
    
    model = nn.Sequential(
        base_model,
        head
    )
    return model


def create_model_v2m(num_classes: int, neuron: int, dropout: float) -> nn.Sequential:
    """
    Cria o modelo EfficientNetV2-M usando a biblioteca torchvision.
    - Congela todas as camadas e descongela os últimos 2 blocos dos features.
    - Remove o classificador original e adiciona uma nova cabeça.
    """
    base_model = models.efficientnet_v2_m(weights=models.EfficientNet_V2_M_Weights.DEFAULT)
    
    for param in base_model.parameters():
        param.requires_grad = False

    for layer in list(base_model.features[-2:]):
        for param in layer.parameters():
            param.requires_grad = True

    # Obter o número de features antes de remover o classificador
    if hasattr(base_model, 'num_features'):
        in_features = base_model.num_features
    elif isinstance(base_model.classifier, nn.Sequential):
        in_features = base_model.classifier[-1].in_features
    else:
        in_features = base_model.classifier.in_features

    base_model.classifier = nn.Identity()

    head = create_head(in_features, neuron, num_classes, dropout)
    
    model = nn.Sequential(
        base_model,
        head
    )
    return model


def create_model_v2l(num_classes: int, neuron: int, dropout: float) -> nn.Sequential:
    """
    Cria o modelo EfficientNetV2-L usando a biblioteca torchvision.
    - Congela todas as camadas e descongela os últimos 2 blocos dos features.
    - Remove o classificador original e adiciona uma nova cabeça.
    """
    base_model = models.efficientnet_v2_l(weights=models.EfficientNet_V2_L_Weights.DEFAULT)
    
    for param in base_model.parameters():
        param.requires_grad = False

    for layer in list(base_model.features[-2:]):
        for param in layer.parameters():
            param.requires_grad = True
            
    # Obter o número de features antes de remover o classificador
    if hasattr(base_model, 'num_features'):
        in_features = base_model.num_features
    elif isinstance(base_model.classifier, nn.Sequential):
        in_features = base_model.classifier[-1].in_features
    else:
        in_features = base_model.classifier.in_features

    base_model.classifier = nn.Identity()

    head = create_head(in_features, neuron, num_classes, dropout)
    
    model = nn.Sequential(
        base_model,
        head
    )
    return model
