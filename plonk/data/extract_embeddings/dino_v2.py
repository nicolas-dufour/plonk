import os, sys

# Ajouter le répertoire racine au chemin
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(root_dir)

import torch
from plonk.utils.image_processing import CenterCrop
from plonk.data.extract_embeddings.dataset_with_path import ImageWithPathDataset
import torch
from torchvision import transforms
from pathlib import Path


from tqdm import tqdm
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--number_of_splits",
    type=int,
    help="Number of splits to process",
    default=1,
)
parser.add_argument(
    "--split_index",
    type=int,
    help="Index of the split to process",
    default=0,
)
parser.add_argument(
    "--input_path",
    type=str,
    help="Path to the input dataset",
)
parser.add_argument(
    "--output_path",
    type=str,
    help="Path to the output dataset",
)

args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14_reg")
model = torch.compile(model, mode="max-autotune")
model.eval()
model.to(device)

input_path = Path(args.input_path)
output_path = Path(args.output_path)

output_path.mkdir(exist_ok=True, parents=True)
augmentation = transforms.Compose(
    [
        CenterCrop(ratio="1:1"),
        transforms.Resize(336, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)
dataset = ImageWithPathDataset(input_path, output_path, transform=augmentation)
dataset = torch.utils.data.Subset(
    dataset,
    range(
        args.split_index * len(dataset) // args.number_of_splits,
        (
            (args.split_index + 1) * len(dataset) // args.number_of_splits
            if args.split_index != args.number_of_splits - 1
            else len(dataset)
        ),
    ),
)

batch_size = 128
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, num_workers=16, collate_fn=lambda x: zip(*x)
)

for images, output_emb_paths in tqdm(dataloader):
    images = torch.stack(images, dim=0).to(device)
    with torch.no_grad():
        embeddings = model(images)
    numpy_embeddings = embeddings.cpu().numpy()
    for emb, output_emb_path in zip(numpy_embeddings, output_emb_paths):
        np.save(f"{output_emb_path}.npy", emb)
