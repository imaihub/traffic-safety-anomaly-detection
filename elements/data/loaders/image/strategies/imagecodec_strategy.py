import imagecodecs
import numpy as np

from elements.data.loaders.image.strategies.base_image_strategy import BaseImageStrategy


class ImageCodecStrategy(BaseImageStrategy):
    @staticmethod
    def get_numpy_array_from_path(path: str) -> np.ndarray:
        img = imagecodecs.imread(path)

        if len(img.shape) == 2 or img.shape[2] == 1:  # Convert to RGB
            img_rgb = np.empty([img.shape[0], img.shape[1], 3], dtype=img.dtype)
            img_rgb[:, :, 0] = img
            img_rgb[:, :, 1] = img
            img_rgb[:, :, 2] = img
        elif len(img.shape) == 3 and img.shape[2] > 3:  # Remove alpha version
            img_rgb = img[:, :, :3]
        else:
            img_rgb = img
        return img_rgb
