import os

import imagecodecs
import numpy as np
import cv2

from elements.data.loaders.image.strategies.imagecodec_strategy import ImageCodecStrategy

strategy_dict = {
    ".png": ImageCodecStrategy,
    ".jpg": ImageCodecStrategy,
    ".jpeg": ImageCodecStrategy,
    ".bmp": ImageCodecStrategy,
    ".tif": ImageCodecStrategy,
    ".tiff": ImageCodecStrategy,
    ".gif": ImageCodecStrategy,
    ".webp": ImageCodecStrategy,
}


def load_mono_image(filename: str, keep_channel: bool = False):
    img = imagecodecs.imread(filename)

    # replicate skimage.io.imread w/ 'as_gray=True'
    # i.e. if image is multichannel, convert it to greyscale and normalise it as float64
    # somehow monochrome images aren't normalised
    if len(img.shape) > 2:
        if img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = np.expand_dims(img, axis=-1)  # restore channel dim for now

        elif img.shape[2] != 1:
            img = np.mean(img, axis=2)

        if not keep_channel:
            # remove channel dimension
            img = img[..., 0]

        img_orig_dtype = img.dtype
        img = img.astype(np.float64)
        if np.issubdtype(img_orig_dtype, np.integer):
            img /= np.iinfo(img_orig_dtype).max

    elif keep_channel:
        # add channel dimension
        img = np.expand_dims(img, -1)

    return img


def load_image(filename: str):
    try:
        file_name, extension = os.path.splitext(filename)
        image = strategy_dict[extension].get_numpy_array_from_path(filename)
        return image
    except Exception as e:
        print(e)
        return None
