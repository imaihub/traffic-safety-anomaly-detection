import numpy as np


def load_npy(filename: str):
    image = np.load(filename)
    if len(image.shape) < 2 or len(image.shape) > 4:
        raise ValueError(f"Cannot understand {len(image.shape)} dimensions.")

    if len(image.shape) == 4:  # it has an extra minibatch dimension
        if image.shape[0] == 1:
            image = image[0]
        else:
            raise ValueError(f"Cannot load 4 dimensions if the first does not have length 1.")

    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=2)

    # auto-detect conversion from c,h,w to h,w,c
    if (image.shape[0] == 1 or image.shape[0] == 3) and (image.shape[2] != 1 and image.shape[2] != 3):
        image = np.moveaxis(image, 0, 2)

    if image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    else:
        if image.shape[2] != 3 and image.shape[2] != 4:
            raise ValueError(f"Cannot understand {image.shape[2]} image channels.")

    return image
