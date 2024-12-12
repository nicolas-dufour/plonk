from transformers import CLIPProcessor


class ClipTransform(object):
    def __init__(self, split):
        self.transform = CLIPProcessor.from_pretrained("geolocal/StreetCLIP")

    def __call__(self, x):
        # return self.transform(images=x, return_tensors="pt")["pixel_values"].squeeze(0)
        return self.transform(images=[x], return_tensors="pt")


if __name__ == "__main__":
    # sanity check
    import glob
    import torchvision.transforms as transforms
    from torchvision.utils import save_image
    from omegaconf import DictConfig, OmegaConf
    from hydra.utils import instantiate
    import torch
    from PIL import Image

    fast_clip_config = OmegaConf.load(
        "./configs/dataset/train_transform/fast_clip.yaml"
    )
    fast_clip_transform = instantiate(fast_clip_config)
    clip_transform = ClipTransform(None)

    img_paths = glob.glob("./datasets/osv5m/test/images/*.jpg")
    original_imgs, re_implemted_imgs, diff = [], [], []

    for i in range(16):
        img = Image.open(img_paths[i])
        clip_img = clip_transform(img)
        fast_clip_img = fast_clip_transform(img)
        original_imgs.append(clip_img)
        re_implemted_imgs.append(fast_clip_img)
        max_diff = (clip_img - fast_clip_img).abs()
        diff.append(max_diff)
        if max_diff.max() > 1e-5:
            print(max_diff.max())
    original_imgs = torch.stack(original_imgs)
    re_implemted_imgs = torch.stack(re_implemted_imgs)
    diff = torch.stack(diff)
