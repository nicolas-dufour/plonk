import torch
import torch.nn.functional as F
import torchvision


def remap_image_torch(image):
    image_torch = ((image + 1) / 2.0) * 255.0
    image_torch = torch.clip(image_torch, 0, 255).to(torch.uint8)
    return image_torch


class CenterCrop(torch.nn.Module):
    """Crops the given image at the center. Allows to crop to the maximum possible size.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        ratio (str): Desired output ratio of the crop that will do the maximum possible crop with the given ratio.
    """

    def __init__(self, size=None, ratio="1:1"):
        super().__init__()
        self.size = size
        self.ratio = ratio

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped.

        Returns:
            PIL Image or Tensor: Cropped image.
        """
        if self.size is None:
            if isinstance(img, torch.Tensor):
                h, w = img.shape[-2:]
            else:
                w, h = img.size
            ratio = self.ratio.split(":")
            ratio = float(ratio[0]) / float(ratio[1])
            ratioed_w = int(h * ratio)
            ratioed_h = int(w / ratio)
            if w >= h:
                if ratioed_h <= h:
                    size = (ratioed_h, w)
                else:
                    size = (h, ratioed_w)
            else:
                if ratioed_w <= w:
                    size = (h, ratioed_w)
                else:
                    size = (ratioed_h, w)
        else:
            size = self.size
        return torchvision.transforms.functional.center_crop(img, size)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size})"
