import torch
import numpy as np
from PIL import Image
import cv2
from config import IMG_SIZE, NORMALIZE_MEAN, NORMALIZE_STD

class GradCAM:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.gradients = None
        self.activations = None
        target_layer = model.conv_head
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        self.model.eval()
        input_tensor = input_tensor.to(self.device).requires_grad_(True)
        output = self.model(input_tensor)
        self.model.zero_grad()
        output[0, class_idx].backward()
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)

        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)

        cam = cam.squeeze().cpu().numpy()
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))

        return cam


def overlay_heatmap(original_image: Image.Image, heatmap: np.ndarray, alpha: float = 0.4) -> Image.Image:
    img = original_image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img)
    heatmap_colored = cv2.applyColorMap(
        (heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    blended = (img_array * (1 - alpha) + heatmap_colored * alpha).astype(np.uint8)

    return Image.fromarray(blended)