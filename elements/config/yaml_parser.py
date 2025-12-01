import yaml
import os


def normalize_path(path: str) -> str:
    """
    Converts a given path to Windows network path if running on Windows,
    otherwise returns the original path.
    """
    path = str(path)
    if os.name == "nt":
        path = path.replace("/", "\\")
    return path


def normalize_paths_in_dict(d: dict) -> dict:
    """
    Recursively normalize all string paths in a dict.
    """
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = normalize_paths_in_dict(v)
        elif isinstance(v, str):
            d[k] = normalize_path(v)
    return d


def get_yaml_dict(file_path: str) -> dict:
    """
    Loads a YAML file and normalizes paths for Windows automatically.
    """
    with open(file_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config = normalize_paths_in_dict(config)
    return config
