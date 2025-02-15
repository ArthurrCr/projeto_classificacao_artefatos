from torchvision import transforms

def get_preprocess_fn(img_height: int = 256, img_width: int = 256):
    """
    Retorna a função de pré-processamento apropriada para o modelo especificado.

    Parâmetros:
    - img_height (int): Altura para redimensionamento das imagens.
    - img_width (int): Largura para redimensionamento das imagens.

    Retorna:
    - transform (torchvision.transforms.Compose): Pipeline de transformações.
    """
    transform = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return transform
