from typing import Any


class InferenceResult:
    def __init__(self, image=None):
        self.image = image
        self.outputs = {}

    def add(self, name: str, value: Any):
        self.outputs[name] = value

    def get(self, name: str):
        return self.outputs.get(name, None)
