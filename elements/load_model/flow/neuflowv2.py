import torch

from elements.data.datatypes.inference_input import InferenceInput
from elements.data.datatypes.inference_result import InferenceResult
from elements.load_model.model_base import BaseModel
from third_party.NeuFlow_v2.NeuFlow.backbone_v7 import ConvBlock
from third_party.NeuFlow_v2.NeuFlow.neuflow import NeuFlow


class NeuFlowV2(BaseModel):
    def __init__(self, weights_path: str, image_width: int, image_height: int):
        self.weights_path = weights_path
        self.image_width = image_width
        self.image_height = image_height
        self.model = self.load_model()

    @torch.no_grad()
    def predict(self, x: InferenceInput) -> InferenceResult:
        sc0, sc1 = x.sc  # unpack the two containers
        img0_tensor = sc0.image_data.get()
        img1_tensor = sc1.image_data.get()

        flow_pred = self.model(img0_tensor, img1_tensor)[-1][0]
        flow_pred = flow_pred.permute(1, 2, 0).cpu().numpy()

        inference_result = InferenceResult(image=sc1.org_image.get())
        inference_result.add("flow", flow_pred)
        return inference_result

    def load_model(self):
        checkpoint_sintel = torch.load(self.weights_path, map_location='cuda')
        device = torch.device('cuda')

        # Sintel
        model_sintel = NeuFlow().to(device)
        model_sintel.load_state_dict(checkpoint_sintel['model'], strict=True)

        for m in model_sintel.modules():
            if type(m) is ConvBlock:
                m.conv1 = self.fuse_conv_and_bn(m.conv1, m.norm1)  # update conv
                m.conv2 = self.fuse_conv_and_bn(m.conv2, m.norm2)  # update conv
                delattr(m, "norm1")  # remove batchnorm
                delattr(m, "norm2")  # remove batchnorm
                m.forward = m.forward_fuse  # update forward

        model_sintel.eval()
        model_sintel.half()

        model_sintel.init_bhwd(1, self.image_height, self.image_width, 'cuda')
        return model_sintel

    def fuse_conv_and_bn(self, conv, bn):
        """Fuse Conv2d() and BatchNorm2d() layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/."""
        fusedconv = (torch.nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=True,
        ).requires_grad_(False).to(conv.weight.device))

        # Prepare filters
        w_conv = conv.weight.clone().view(conv.out_channels, -1)
        w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
        fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

        # Prepare spatial bias
        b_conv = torch.zeros(conv.weight.shape[0], device=conv.weight.device) if conv.bias is None else conv.bias
        b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
        fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

        return fusedconv
