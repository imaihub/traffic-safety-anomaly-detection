import torch
import clip
import numpy as np

from elements.data.datatypes.inference_input import InferenceInput
from elements.data.datatypes.inference_result import InferenceResult

from elements.load_model.model_base import BaseModel


class CLIP(BaseModel):
    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device
        self.model = self.load_model()

    def load_model(self) -> torch.nn.Module:
        self.model, _ = clip.load("ViT-B/16", device=self.device)
        self.model.eval()
        print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in self.model.parameters()]):,}")
        print("Context length:", self.model.context_length)
        print("Vocab size:", self.model.vocab_size)
        return self.model

    @torch.no_grad()
    def predict(self, x: InferenceInput) -> InferenceResult:
        inference_result = InferenceResult(image=x.sc.org_image.get())
        image = x.sc.image_data.get()
        features = self.model.encode_image(image)
        features = features.contiguous().detach().cpu().numpy()
        inference_result.add("features", features)
        return inference_result
