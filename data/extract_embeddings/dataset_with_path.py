from PIL import Image
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm


class ImageWithPathDataset(torch.utils.data.Dataset):
    def __init__(self, root_image_path, output_path, transform=None):
        self.root_image_path = root_image_path
        self.image_paths = list(root_image_path.glob("**/*.jpg"))
        self.transform = transform
        self.output_path = output_path

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        output_emb_path = self.output_path / image_path.parent.relative_to(
            self.root_image_path
        )
        output_emb_path.mkdir(exist_ok=True, parents=True)
        output_emb_path = output_emb_path / image_path.stem
        return image, output_emb_path
