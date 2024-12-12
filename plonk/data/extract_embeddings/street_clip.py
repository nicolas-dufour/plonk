import os, sys

# Ajouter le répertoire racine au chemin
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(root_dir)

from PIL import Image
from pathlib import Path
import torch
from transformers import CLIPProcessor, CLIPVisionModel
import numpy as np
from tqdm import tqdm
from plonk.data.extract_embeddings.dataset_with_path import ImageWithPathDataset


device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPVisionModel.from_pretrained("geolocal/StreetCLIP").to(device)
processor = CLIPProcessor.from_pretrained("geolocal/StreetCLIP")

input_path = Path("datasets/osv5m/images")
output_path = Path("datasets/osv5m/embeddings/street_clip")

output_path.mkdir(exist_ok=True, parents=True)

dataset = ImageWithPathDataset(input_path)

batch_size = 128
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, num_workers=16, collate_fn=lambda x: zip(*x)
)

for images, output_emb_paths in tqdm(dataloader):
    inputs = processor(images=images, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0]
    numpy_embeddings = embeddings.cpu().numpy()
    for emb, output_emb_path in zip(numpy_embeddings, output_emb_paths):
        np.save(f"{output_emb_path}.npy", emb)
