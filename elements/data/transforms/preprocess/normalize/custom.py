import torch

from elements.data.transforms.preprocess.normalize.enums import NormalizeValues


class NormalizeCustom:
    def __init__(self, mean: list | NormalizeValues, std: list | NormalizeValues):
        super().__init__()
        if isinstance(mean, NormalizeValues):
            self.mean = mean.value
        if isinstance(std, NormalizeValues):
            self.std = std.value
        self.mean = torch.tensor(self.mean, dtype=torch.float32)
        self.std = torch.tensor(self.std, dtype=torch.float32)

    def __call__(self, image: torch.Tensor):
        device = image.device

        if image.ndim == 3:
            # CHW or HWC
            if image.shape[0] == 3:
                # CHW format
                mean = self.mean.view(3, 1, 1).to(device)
                std = self.std.view(3, 1, 1).to(device)
                image = (image - mean) / std
            elif image.shape[2] == 3:
                # HWC format
                mean = self.mean.view(1, 1, 3).to(device)
                std = self.std.view(1, 1, 3).to(device)
                image = (image - mean) / std
            else:
                raise ValueError("Invalid 3D image format: must be CHW or HWC with 3 channels.")
        elif image.ndim == 4:
            # BCHW
            if image.shape[1] == 3:
                mean = self.mean.view(1, 3, 1, 1).to(device)
                std = self.std.view(1, 3, 1, 1).to(device)
                image = (image - mean) / std
            else:
                raise ValueError("Expected BCHW format with 3 channels.")

        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")

        return image


class DenormalizeCustom:
    def __init__(self, mean: list = None, std: list = None):
        super().__init__()
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

    def __call__(self, image: torch.Tensor):
        device = image.device

        if image.ndim == 3:
            # CHW or HWC
            if image.shape[0] == 3:
                mean = self.mean.view(3, 1, 1).to(device)
                std = self.std.view(3, 1, 1).to(device)
                image = image * std + mean
            elif image.shape[2] == 3:
                mean = self.mean.view(1, 1, 3).to(device)
                std = self.std.view(1, 1, 3).to(device)
                image = image * std + mean
            else:
                raise ValueError("Invalid 3D image format: must be CHW or HWC with 3 channels.")

        elif image.ndim == 4:
            if image.shape[1] == 3:
                mean = self.mean.view(1, 3, 1, 1).to(device)
                std = self.std.view(1, 3, 1, 1).to(device)
                image = image * std + mean
            else:
                raise ValueError("Expected BCHW format with 3 channels.")

        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")

        return image
