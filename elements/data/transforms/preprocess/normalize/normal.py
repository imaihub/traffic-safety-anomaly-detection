import torch


class Normalize:
    """
    Normalize an image by scaling pixel values from [0, 255] to [0.0, 1.0].
    Supports BCHW, CHW, and HWC.
    """
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        Normalize pixel values to [0.0, 1.0].
        """
        if image.dtype != torch.float32:
            image = image.to(torch.float32)

        return image / 255.0


class Denormalize:
    """
    Denormalize an image by scaling pixel values from [0.0, 1.0] to [0, 255].
    Supports BCHW, CHW, and HWC.
    """
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        Denormalize pixel values to [0, 255].
        """
        image = image * 255.0
        return image.clamp(0, 255).to(torch.uint8)
