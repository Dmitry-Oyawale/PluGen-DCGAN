import torch
import torchvision.utils as vutils


def save_image_grid(img_tensor: torch.Tensor, out_path: str, nrow: int = 8):
    """
    img_tensor: [B, 3, 64, 64] in [-1, 1] (because Normalize/Tanh)
    """
    vutils.save_image(img_tensor.detach().cpu(), out_path, normalize=True, nrow=nrow)

