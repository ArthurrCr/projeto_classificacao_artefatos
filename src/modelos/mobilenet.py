import torch.nn as nn
import timm

def create_head(in_features: int, neuron: int, num_classes: int, dropout: float) -> nn.Sequential:
    """
    Cria a nova cabeça para o modelo.
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

def create_model_mobilenetv2(num_classes: int, neuron: int, dropout: float) -> nn.Sequential:
    # Cria o modelo base com mobilenetv2_100 pré-treinado
    base_model = timm.create_model('mobilenetv2_100', pretrained=True)

    # Congelar todas as camadas inicialmente
    for param in base_model.parameters():
        param.requires_grad = False

    # Descongelar os últimos 2 blocos
    for block in base_model.blocks[-2:]:
        for param in block.parameters():
            param.requires_grad = True

    # Verificar se o modelo possui AdaptiveAvgPool e, se não, forçar o uso dele
    if not isinstance(base_model.global_pool, nn.AdaptiveAvgPool2d):
        base_model.global_pool = nn.AdaptiveAvgPool2d(1)

    # Remove o classificador original
    base_model.classifier = nn.Identity()

    # Obter o número de features de saída do base model
    in_features = base_model.num_features

    # Cria a nova cabeça
    head = create_head(in_features, neuron, num_classes, dropout)

    # Adiciona um Flatten caso a saída do base_model seja (batch, features, 1, 1)
    model = nn.Sequential(
        base_model,
        nn.Flatten(1),  # Achata a partir da dimensão 1, transformando [batch, features, 1, 1] em [batch, features]
        head
    )
    return model
