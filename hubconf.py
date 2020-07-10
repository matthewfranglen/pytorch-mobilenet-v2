"""
TORCH.HUB configuration for pytorch-mobilenet-v2
See https://pytorch.org/docs/stable/hub.html
Use with:
    model = torch.hub.load('matthewfranglen/pytorch-mobilenet-v2', 'mobilenet_v2', pretrained=True)
"""

dependencies = ["torch"]

from MobileNetV2 import mobilenet_v2  # pylint: disable=wrong-import-position

__all__ = ["mobilenet_v2"]
