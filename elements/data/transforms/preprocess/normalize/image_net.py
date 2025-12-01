import torch


class NormalizeImageNet:
    """
    Normalize an image using the basic ImageNet mean and std.
    Supports CHW, HWC, and BCHW formats.
    """
    def __init__(self) -> None:
        super().__init__()
        self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
        self.std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        device = image.device

        if image.ndim == 3:
            # CHW or HWC
            if image.shape[0] == 3:
                # CHW
                mean = self.mean.view(3, 1, 1).to(device)
                std = self.std.view(3, 1, 1).to(device)
            elif image.shape[2] == 3:
                # HWC
                mean = self.mean.view(1, 1, 3).to(device)
                std = self.std.view(1, 1, 3).to(device)
            else:
                raise ValueError("Expected 3 channels in CHW or HWC format.")

        elif image.ndim == 4:
            # BCHW
            if image.shape[1] == 3:
                mean = self.mean.view(1, 3, 1, 1).to(device)
                std = self.std.view(1, 3, 1, 1).to(device)
            else:
                raise ValueError("Expected 3 channels in BCHW format.")

        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")

        image = (image - mean) / std
        return image


class DenormalizeImageNet:
    """
    Denormalize an image using the basic ImageNet mean and std.
    Supports CHW, HWC, and BCHW formats.
    """
    def __init__(self) -> None:
        super().__init__()
        self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
        self.std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        device = image.device

        if image.ndim == 3:
            if image.shape[0] == 3:
                mean = self.mean.view(3, 1, 1).to(device)
                std = self.std.view(3, 1, 1).to(device)
            elif image.shape[2] == 3:
                mean = self.mean.view(1, 1, 3).to(device)
                std = self.std.view(1, 1, 3).to(device)
            else:
                raise ValueError("Expected 3 channels in CHW or HWC format.")

        elif image.ndim == 4:
            if image.shape[1] == 3:
                mean = self.mean.view(1, 3, 1, 1).to(device)
                std = self.std.view(1, 3, 1, 1).to(device)
            else:
                raise ValueError("Expected 3 channels in BCHW format.")

        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")

        image = image * std + mean
        return image
