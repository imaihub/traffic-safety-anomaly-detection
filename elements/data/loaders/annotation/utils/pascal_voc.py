import numpy as np
from skimage import io


def load_label_map(label_map_path, background='background'):
    label_map = dict()
    label_idx = 1
    with open(label_map_path, 'r') as lf:
        # Skip first line, has the header
        lf.readline()
        line = lf.readline().strip()
        while line:
            line_arr = line.split(':')
            if line_arr[0] != background:
                # Save reference color and index value
                label_map[line_arr[0]] = (np.array([int(x) for x in line_arr[1].split(',')]), label_idx)

                label_idx += 1

            # Read next line
            line = lf.readline().strip()
    return label_map


def load_class_mask_pascal(image_fpath: str, label_map: dict):
    image = io.imread(image_fpath)

    # Create blank mask
    mask = np.zeros(image.shape[:2], dtype='uint8')

    for _, (c, idx) in label_map.items():
        # Add index-label for each pixel
        mask += np.all(image == c, axis=-1).astype('uint8') * idx

    return mask
