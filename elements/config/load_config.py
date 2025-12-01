import os

import yaml

from elements.config.yaml_parser import get_yaml_dict


class Config:
    def __init__(self, path: str = os.path.join("core", "config", "config.yml")):
        self.config = get_yaml_dict(path)
        self.path = path

    def get(self, name: str):
        return self.config.get(name, None)

    def add_field(self, name: str, value: object, write: bool = True):
        if name in self.config:
            self.config[name].update(value)
        else:
            self.config[name] = value

        if write:
            self.write()

    def __enter__(self):
        return self

    def write(self):
        with open(self.path, "w", newline="", encoding="utf-8") as yaml_file:
            yaml.dump(self.config, yaml_file, default_flow_style=False)

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        self.write()
