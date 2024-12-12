"""
Adapted from https://github.com/nv-nguyen/template-pose/blob/main/src/utils/augmentation.py
"""

from torchvision import transforms
from PIL import ImageEnhance, ImageFilter, Image
import numpy as np
import random
import logging
from torchvision.transforms import RandomResizedCrop, ToTensor


class PillowRGBAugmentation:
    def __init__(self, pillow_fn, p, factor_interval):
        self._pillow_fn = pillow_fn
        self.p = p
        self.factor_interval = factor_interval

    def __call__(self, PIL_image):
        if random.random() <= self.p:
            factor = random.uniform(*self.factor_interval)
            if PIL_image.mode != "RGB":
                logging.warning(
                    f"Error when apply data aug, image mode: {PIL_image.mode}"
                )
                imgs = imgs.convert("RGB")
                logging.warning(f"Success to change to {PIL_image.mode}")
            PIL_image = (self._pillow_fn(PIL_image).enhance(factor=factor)).convert(
                "RGB"
            )
        return PIL_image


class PillowSharpness(PillowRGBAugmentation):
    def __init__(
        self,
        p=0.3,
        factor_interval=(0, 40.0),
    ):
        super().__init__(
            pillow_fn=ImageEnhance.Sharpness,
            p=p,
            factor_interval=factor_interval,
        )


class PillowContrast(PillowRGBAugmentation):
    def __init__(
        self,
        p=0.3,
        factor_interval=(0.5, 1.6),
    ):
        super().__init__(
            pillow_fn=ImageEnhance.Contrast,
            p=p,
            factor_interval=factor_interval,
        )


class PillowBrightness(PillowRGBAugmentation):
    def __init__(
        self,
        p=0.5,
        factor_interval=(0.5, 2.0),
    ):
        super().__init__(
            pillow_fn=ImageEnhance.Brightness,
            p=p,
            factor_interval=factor_interval,
        )


class PillowColor(PillowRGBAugmentation):
    def __init__(
        self,
        p=1,
        factor_interval=(0.0, 20.0),
    ):
        super().__init__(
            pillow_fn=ImageEnhance.Color,
            p=p,
            factor_interval=factor_interval,
        )


class PillowBlur:
    def __init__(self, p=0.4, factor_interval=(1, 3)):
        self.p = p
        self.k = random.randint(*factor_interval)

    def __call__(self, PIL_image):
        if random.random() <= self.p:
            PIL_image = PIL_image.filter(ImageFilter.GaussianBlur(self.k))
        return PIL_image


class NumpyGaussianNoise:
    def __init__(self, p, factor_interval=(0.01, 0.3)):
        self.noise_ratio = random.uniform(*factor_interval)
        self.p = p

    def __call__(self, img):
        if random.random() <= self.p:
            img = np.copy(img)
            noisesigma = random.uniform(0, self.noise_ratio)
            gauss = np.random.normal(0, noisesigma, img.shape) * 255
            img = img + gauss

            img[img > 255] = 255
            img[img < 0] = 0
        return Image.fromarray(np.uint8(img))


class StandardAugmentation:
    def __init__(
        self, names, brightness, contrast, sharpness, color, blur, gaussian_noise
    ):
        self.brightness = brightness
        self.contrast = contrast
        self.sharpness = sharpness
        self.color = color
        self.blur = blur
        self.gaussian_noise = gaussian_noise

        # define a dictionary of augmentation functions to be applied
        self.names = names.split(",")
        self.augmentations = {
            "brightness": self.brightness,
            "contrast": self.contrast,
            "sharpness": self.sharpness,
            "color": self.color,
            "blur": self.blur,
            "gaussian_noise": self.gaussian_noise,
        }

    def __call__(self, img):
        for name in self.names:
            img = self.augmentations[name](img)
        return img


class GeometricAugmentation:
    def __init__(
        self,
        names,
        random_resized_crop,
        random_horizontal_flip,
        random_vertical_flip,
        random_rotation,
    ):
        self.random_resized_crop = random_resized_crop
        self.random_horizontal_flip = random_horizontal_flip
        self.random_vertical_flip = random_vertical_flip
        self.random_rotation = random_rotation
        self.names = names.split(",")

        self.augmentations = {
            "random_resized_crop": self.random_resized_crop,
            "random_horizontal_flip": self.random_horizontal_flip,
            "random_vertical_flip": self.random_vertical_flip,
            "random_rotation": self.random_rotation,
        }

    def __call__(self, img):
        for name in self.names:
            img = self.augmentations[name](img)
        return img


class ImageAugmentation:
    def __init__(
        self, names, clip_transform, standard_augmentation, geometric_augmentation
    ):
        self.clip_transform = clip_transform
        self.standard_augmentation = standard_augmentation
        self.geometric_augmentation = geometric_augmentation
        self.names = names.split(",")
        self.transforms = {
            "clip_transform": self.clip_transform,
            "standard_augmentation": self.standard_augmentation,
            "geometric_augmentation": self.geometric_augmentation,
        }
        print(f"Image augmentation: {self.names}")

    def __call__(self, img):
        for name in self.names:
            img = self.transforms[name](img)
        return img


if __name__ == "__main__":
    # sanity check
    import glob
    import torchvision.transforms as transforms
    from torchvision.utils import save_image
    from omegaconf import DictConfig, OmegaConf
    from hydra.utils import instantiate
    import torch
    from PIL import Image

    augmentation_config = OmegaConf.load(
        "./configs/dataset/train_transform/augmentation.yaml"
    )
    augmentation_config.names = "standard_augmentation,geometric_augmentation"
    augmentation_transform = instantiate(augmentation_config)
    img_paths = glob.glob("./datasets/osv5m/test/images/*.jpg")

    num_try = 20
    num_try_per_image = 8
    num_imgs = 8

    for idx in range(num_try):
        imgs = []
        for idx_img in range(num_imgs):
            img = Image.open(img_paths[idx_img])
            for idx_try in range(num_try_per_image):
                if idx_try == 0:
                    imgs.append(ToTensor()(img.resize((224, 224))))
                img_aug = augmentation_transform(img.copy())
                img_aug = ToTensor()(img_aug)
                imgs.append(img_aug)
        imgs = torch.stack(imgs)
        save_image(imgs, f"augmentation_{idx:03d}.png", nrow=9)
