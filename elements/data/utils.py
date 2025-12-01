import glob
import os
import platform
from typing import Optional

import natsort
import numpy as np

from elements.enums.enums import FileExtension, all_image_extensions


def get_datasets_root_dir(private: bool = False):
    """
    Get the root directory where the datasets are stored.

    :param private: if True, returns the private_data datasets root, public_data datasets root otherwise.
    :return: root of datasets directory.
    """

    p = platform.system()
    if not private:
        if p == "Windows":
            return os.path.join("x:", "datasets")
        elif p == "Linux":
            return os.path.join(os.sep, "media", "public_data", "datasets")
        else:
            raise RuntimeError(f"Unrecognized unsupported platform {platform.platform()}")
    else:
        if p == "Linux":
            return os.path.join(os.sep, "media", "private_data", "MasterCVDS", "datasets")
        else:
            raise RuntimeError(f"Unrecognized or unsupported platform {platform.platform()}")


def count_images(dataset_path: str, extensions: list[FileExtension]) -> int:
    frames = []
    for extension in extensions:
        frames.extend(glob.glob(os.path.join(dataset_path, "**", f'*.{extension.value}'), recursive=True))
    return len(frames)


def prune_lists_smallest_length(lists: list[list]):
    min_count = min([len(l) for l in lists])
    print(f"Pruning lists to length of {min_count}...")
    return [l[:min_count] for l in lists]


def concatenate_flatten_arrays(arrays: Optional[list[np.ndarray]] = None, file_paths: Optional[list[str]] = None):
    if arrays is not None:
        pass
    elif file_paths is not None:
        arrays = [np.load(f, allow_pickle=True).astype(np.int32).flatten() for f in file_paths]
    array = np.concatenate(arrays, 0)
    return array


def get_file_path(file_path: str, directory: str, extension: FileExtension, subfolder_keep_count: int = 0) -> str:
    if subfolder_keep_count > 0:
        os.makedirs(os.path.join(directory, os.sep.join(os.path.dirname(file_path).split(os.sep)[-subfolder_keep_count:])), exist_ok=True)
        return os.path.join(directory, os.sep.join(os.path.dirname(file_path).split(os.sep)[-subfolder_keep_count:]), os.path.basename(file_path).split(".")[0] + (f"{extension.value}" if extension.value.startswith('.') else f'.{extension.value}'))
    return os.path.join(directory, os.path.basename(file_path).split(".")[0] + (f"{extension.value}" if extension.value.startswith('.') else f'.{extension.value}'))


def ensure_dir_exists(dirs: list[str]):
    for d in dirs:
        if not os.path.isdir(d):
            print(f"Creating directory {d} as it does not exist")
            os.makedirs(d, exist_ok=True)


def get_file_paths(directory: str, extensions: list[FileExtension] | list[str], necessary_dir_words: Optional[list[str]] = None, recursive: bool = False, sort_list: bool = True, max_count: int = -1) -> list[str]:
    """
    Get All file paths based on root directory, extensions and a recursive flag

    :param directory: Root directory
    :param extensions: List of file extensions to include
    :param recursive: Recursive flag
    :param necessary_dir_words: Necessary words to be in path, a whitelist
    :param sort_list: Whether to sort the final list or not
    :param max_count: Maximum number of paths to load

    :return: List of file paths
    """
    file_paths = []
    if isinstance(extensions[0], FileExtension):  # Assume the rest is of type FileExtension as well
        extensions = [extension.value for extension in extensions]

    def _recursive_collect(dir: str):
        nonlocal file_paths
        # Sort directories and files for consistent order
        with os.scandir(dir) as entries:
            dirs = []
            files = []
            for entry in entries:
                if entry.is_dir():
                    dirs.append(entry.name)
                else:
                    files.append(entry.name)
            # Sort directories and files
            dirs = natsort.natsorted(dirs)
            files = natsort.natsorted(files)
            # Process files
            for file in files:
                if os.path.splitext(file)[1][1:] in extensions:
                    if necessary_dir_words is not None:
                        for necessary_dir_word in necessary_dir_words:
                            if necessary_dir_word not in file:
                                continue
                    file_paths.append(os.path.join(dir, file))
                    if max_count > 0 and len(file_paths) >= max_count:
                        return
            # Recurse into directories
            for subdir in dirs:
                _recursive_collect(os.path.join(dir, subdir))
                if max_count > 0 and len(file_paths) >= max_count:
                    return

    if max_count < 0:
        if recursive:
            for extension in extensions:
                file_paths.extend(glob.glob(os.path.join(directory, "**", f'*{extension}'), recursive=True))
        else:
            for extension in extensions:
                file_paths.extend(glob.glob(os.path.join(directory, f'*{extension}'), recursive=False))
    elif max_count > 0:
        _recursive_collect(directory)
    if sort_list:
        file_paths = natsort.natsorted(file_paths)
    return file_paths


def get_image_paths(path: str | list[str], extensions: list[str] | list[FileExtension] = all_image_extensions):
    if isinstance(extensions[0], FileExtension):  # Assume the rest is of type FileExtension as well
        extensions = [extension.value for extension in extensions]

    if isinstance(path, str):  # Either list of files or path to directory
        if os.path.isdir(path):
            # Collect in sorted list
            return get_file_paths(directory=path, recursive=True, sort_list=True, extensions=extensions)
        elif os.path.isfile(path):
            ext = os.path.splitext(path)[1]
            if ext not in extensions:  # Assume text file
                return read_lines(file_path=path)
            else:  # It is one image path:
                return [path]
        else:
            raise Exception("Could not find input files")
    elif isinstance(path, list):  # Already in the correct format
        return path

    raise Exception("Could not find input files")


def read_lines(file_path: str) -> list[str]:
    paths = []
    with open(file_path, "r") as f:
        paths = f.readlines()
        paths = [path.strip() for path in paths]

    return paths
