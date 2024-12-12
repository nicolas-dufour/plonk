import webdataset as wds
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from plonk.utils.image_processing import CenterCrop
from tqdm import tqdm
import os

tqdm.pandas()

print("Loading dinov2")
augmentation_dinov2 = transforms.Compose(
    [
        CenterCrop(ratio="1:1"),
        transforms.Resize(336, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14_reg")
model.eval()
model.to(device)
print(f"Model loaded on {device}")


class YFCCDataset(Dataset):
    def __init__(self, csv_path, images_root):
        self.df = pd.read_csv(csv_path, sep="\t")
        self.df = self.df[self.df["latitude"].notna() & self.df["longitude"].notna()]
        self.images_root = Path(images_root)

        # Create image paths and check existence
        print("Checking image existence...")
        self.df["image_path"] = self.df["hash"].progress_apply(
            lambda x: self.images_root / x[:3] / x[3:6] / f"{x}.jpg"
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row["image_path"]

        if not image_path.exists():
            print(f"Image {image_path} does not exist")
            return None

        # Read the JPEG file directly as bytes
        with open(image_path, "rb") as f:
            jpg_data = f.read()

        image = Image.open(image_path).convert("RGB")
        image = augmentation_dinov2(image)

        # Convert metadata to dict and ensure all values are JSON serializable
        metadata = row.to_dict()
        del metadata["image_path"]

        return {
            "image": image,
            "jpg_data": jpg_data,
            "photo_id": str(row["photo_id"]),
            "metadata": metadata,
        }


def custom_collate(batch):
    """
    Custom collate function to handle dictionary items from the dataset
    """
    return {
        "image": torch.stack([item["image"] for item in batch if item is not None]),
        "jpg_data": [item["jpg_data"] for item in batch if item is not None],
        "photo_id": [item["photo_id"] for item in batch if item is not None],
        "metadata": [item["metadata"] for item in batch if item is not None],
    }


def process_batch(batch, model, device):
    images = batch["image"].to(device)  # No need to stack, already stacked in collate
    with torch.no_grad():
        embeddings = model(images).cpu().numpy()

    samples = []
    for i in range(len(batch["photo_id"])):
        sample = {
            "__key__": batch["photo_id"][i],
            "jpg": batch["jpg_data"][i],
            "dinov2_vitl14_registers.npy": embeddings[i],
            "json": batch["metadata"][i],
        }
        samples.append(sample)
    return samples


def main(
    src_csv,
    src_images,
    dest_folder,
    num_samples_per_tar=10000,
    job_offset=0,
    batch_size=32,
):
    print(f"Loading dataset")
    dataset = YFCCDataset(src_csv, src_images)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        collate_fn=custom_collate,  # Add the custom collate function
    )

    print(f"Processing job {job_offset} with {len(dataset)} samples")
    with wds.ShardWriter(
        str(Path(dest_folder) / "%04d.tar"),
        maxcount=num_samples_per_tar,
        start_shard=10 * job_offset,
    ) as sink:
        for batch in tqdm(dataloader):
            samples = process_batch(batch, model, device)
            for sample in samples:
                sink.write(sample)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--src_csv_dir", help="pixel_input_folder")
    parser.add_argument("--src_images_dir", help="path to source images")
    parser.add_argument("--dest", help="path to destination web")
    parser.add_argument(
        "--num_samples_per_tar",
        help="number of samples per tar",
        type=int,
        default=10000,
    )
    parser.add_argument("--job_offset", help="job offset", type=int, default=0)
    parser.add_argument("--batch_size", help="batch size", type=int, default=256)
    args = parser.parse_args()

    dest = Path(args.dest)
    dest.mkdir(exist_ok=True, parents=True)

    main(
        Path(args.src_csv_dir) / f"{str(args.job_offset).zfill(3)}.csv",
        args.src_images_dir,
        args.dest,
        args.num_samples_per_tar,
        args.job_offset,
        args.batch_size,
    )
