from torchvision import transforms

def get_augmentation_fn(
    img_height: int = 256,
    img_width: int = 256,
    mean=None,
    std=None,
    # Parâmetros de rotação
    rotation_degrees: float = 0.0, 
    
    # Parâmetros de color jitter
    apply_color_jitter: bool = False,
    brightness: float = 0.0,
    contrast: float = 0.0,
    saturation: float = 0.0,
    hue: float = 0.0,
    
    # Redimensionamento e crop
    apply_random_resized_crop: bool = False,
    scale: tuple = (0.8, 1.0),
    
    # Flip horizontal
    apply_horizontal_flip: bool = False,
    horizontal_flip_prob: float = 0.5,
    
    # Zoom
    apply_zoom: bool = False,
    zoom_factor: float = 1.0,  # 1.0 = sem zoom
    zoom_prob: float = 0.5,    # Probabilidade de aplicar o zoom
    
    # Gaussian Blur
    apply_gaussian_blur: bool = False,
    gaussian_blur_kernel: int = 3,
    gaussian_blur_prob: float = 0.5,
    gaussian_blur_sigma: float = 1.0
):
    """
    Retorna um pipeline de transformações com base nos parâmetros fornecidos.
    Todos os parâmetros possuem valores padrão neutros, de forma que se não
    forem modificados, nenhuma transformação será aplicada além do ToTensor
    e da normalização.
    """

    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]

    transform_list = []

    # Aplica RandomResizedCrop se solicitado
    if apply_random_resized_crop:
        transform_list.append(transforms.RandomResizedCrop(
            (img_height, img_width), scale=scale
        ))
    else:
        # Caso contrário, redimensiona a imagem para o tamanho desejado
        transform_list.append(transforms.Resize((img_height, img_width)))

    # Aplica rotação se rotation_degrees > 0
    if rotation_degrees > 0:
        transform_list.append(transforms.RandomRotation(degrees=rotation_degrees))

    # Aplica color jitter se solicitado
    if apply_color_jitter and (brightness > 0 or contrast > 0 or saturation > 0 or hue > 0):
        transform_list.append(transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue
        ))
    
    # Aplica zoom se solicitado, com probabilidade
    if apply_zoom and zoom_factor != 1.0:
        zoom_transform = transforms.RandomAffine(
            degrees=0,  # Sem rotação
            scale=(zoom_factor, zoom_factor)  # Zoom fixo
        )
        transform_list.append(transforms.RandomApply(
            [zoom_transform],
            p=zoom_prob  # Probabilidade de aplicar o zoom
        ))

    # Aplica flip horizontal se solicitado
    if apply_horizontal_flip and horizontal_flip_prob > 0:
        transform_list.append(transforms.RandomHorizontalFlip(p=horizontal_flip_prob))
    
    # Aplica blur gaussiano se solicitado
    if apply_gaussian_blur and gaussian_blur_prob > 0:
        transform_list.append(transforms.RandomApply(
            [transforms.GaussianBlur(
                kernel_size=gaussian_blur_kernel, 
                sigma=(gaussian_blur_sigma, gaussian_blur_sigma)
            )],
            p=gaussian_blur_prob
        ))
    
    # Converte para tensor
    transform_list.append(transforms.ToTensor())
    # Normaliza
    transform_list.append(transforms.Normalize(mean=mean, std=std))
    
    augmentation_transform = transforms.Compose(transform_list)
    return augmentation_transform
