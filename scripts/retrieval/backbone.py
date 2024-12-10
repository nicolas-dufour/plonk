from os.path import join
import PIL
import numpy as np
import pandas as pd
import reverse_geocoder
from torch.utils.data import Dataset


class GeoDataset(Dataset):
    def __init__(self, image_folder, annotation_file, transformation, tag="image_id"):
        self.image_folder = image_folder
        gt = pd.read_csv(annotation_file, dtype={tag: str})
        files = set([f.replace(".jpg", "") for f in os.listdir(image_folder)])
        gt = gt[gt[tag].isin(files)]
        self.processor = transformation
        self.gt = [
            (g[1][tag], g[1]["latitude"], g[1]["longitude"]) for g in gt.iterrows()
        ]
        self.tag = tag

    def fid(self, i):
        return self.gt[i][0]

    def latlon(self, i):
        return self.gt[i][1]

    def __len__(self):
        return len(self.gt)

    def __getitem__(self, idx):
        fp = join(self.image_folder, self.gt[idx][0] + ".jpg")
        return self.processor(self, idx, fp)


def load_plonk(path):
    import hydra
    from hydra import initialize, compose
    from models.module import DiffGeolocalizer
    from omegaconf import OmegaConf, open_dict
    from os.path import join
    from hydra.utils import instantiate

    # load config from path
    # make path relative to current_dir
    with initialize(version_base=None, config_path="osv5m__best_model"):
        cfg = compose(config_name="config", overrides=[])

    checkpoint = torch.load(join(path, "last.ckpt"))
    del checkpoint["state_dict"][
        "model.backbone.clip.vision_model.embeddings.position_ids"
    ]
    torch.save(checkpoint, join(path, "last2.ckpt"))

    with open_dict(cfg):
        cfg.checkpoint = join(path, "last2.ckpt")

    cfg.num_classes = 11399
    cfg.model.network.mid.instance.final_dim = cfg.num_classes * 3
    cfg.model.network.head.final_dim = cfg.num_classes * 3
    cfg.model.network.head.instance.quadtree_path = join(path, "quadtree_10_1000.csv")

    cfg.dataset.train_dataset.path = ""
    cfg.dataset.val_dataset.path = ""
    cfg.dataset.test_dataset.path = ""
    cfg.logger.save_dir = ""
    cfg.data_dir = ""
    cfg.root_dir = ""
    cfg.mode = "test"
    cfg.model.network.backbone.instance.path = (
        "laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K"
    )
    transform = instantiate(cfg.dataset.test_transform)
    model = DiffGeolocalizer.load_from_checkpoint(
        join(path, "last2.ckpt"), cfg=cfg.model
    )
    os.remove(join(path, "last2.ckpt"))

    @torch.no_grad()
    def inference(model, x):
        return x[0], model.model.backbone({"img": x[1].to(model.device)})[:, 0, :].cpu()

    def collate_fn(batch):
        return [b[0] for b in batch], torch.stack([b[1] for b in batch], dim=0)

    def operate(self, idx, fp):
        proc = self.processor(PIL.Image.open(fp))
        return self.gt[idx][0], proc

    return model, operate, inference, collate_fn


def load_clip(which):
    # We evaluate on:
    # - "openai/clip-vit-base-patch32"
    # - "openai/clip-vit-large-patch14-336"
    # - "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
    # - "laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K"
    # - "geolocal/StreetCLIP"
    from transformers import CLIPProcessor, CLIPModel

    @torch.no_grad()
    def inference(model, img):
        image_ids = img.data.pop("image_id")
        image_input = img.to(model.device)
        image_input["pixel_values"] = image_input["pixel_values"].squeeze(1)
        features = model.get_image_features(**image_input)
        features /= features.norm(dim=-1, keepdim=True)
        return image_ids, features.cpu()

    processor = CLIPProcessor.from_pretrained(which)

    def operate(self, idx, fp):
        pil = PIL.Image.open(fp)
        proc = processor(images=pil, return_tensors="pt")
        proc["image_id"] = self.gt[idx][0]
        return proc

    return CLIPModel.from_pretrained(which), operate, inference, None


def load_dino(which):
    # We evaluate on:
    # - 'facebook/dinov2-large'
    from transformers import AutoImageProcessor, AutoModel

    @torch.no_grad()
    def inference(model, img):
        image_ids = img.data.pop("image_id")
        image_input = img.to(model.device)
        image_input["pixel_values"] = image_input["pixel_values"].squeeze(1)
        features = model(**image_input).last_hidden_state[:, 0]
        features /= features.norm(dim=-1, keepdim=True)
        return image_ids, features.cpu()

    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-large")

    def operate(self, idx, fp):
        pil = PIL.Image.open(fp)
        proc = processor(images=pil, return_tensors="pt")
        proc["image_id"] = self.gt[idx][0]
        return proc

    return AutoModel.from_pretrained("facebook/dinov2-large"), operate, inference, None


def get_backbone(name):
    if os.path.isdir(name):
        return load_plonk(name)
    elif "clip" in name.lower():
        return load_clip(name)
    elif "dino" in name.lower():
        return load_dino(name)
