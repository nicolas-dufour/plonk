import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import argparse
import json
from collections import UserDict
from pathlib import Path

import numpy as np
import torch
import webdataset as wds
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from webdataset.autodecode import ImageHandler
from plonk.utils.image_processing import CenterCrop

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

dinov2_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14_reg")
dinov2_model.eval()
dinov2_model.to(device)
print(f"Model loaded on {device}")


def dict_collate(batch):
    output_dict = {}
    if isinstance(batch[0], dict):
        for key in batch[0].keys():
            list_key = [d[key] for d in batch]
            if key != "json":
                output_dict[key] = dict_collate(list_key)
            else:
                output_dict[key] = list_key
        return output_dict
    elif isinstance(batch[0], Image.Image):
        return [img for img in batch]
    else:
        return torch.utils.data.dataloader.default_collate(batch)


def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning, and continue."""
    # logging.warning(f"Handling webdataset error ({repr(exn)}). Ignoring.")
    return True


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def add_clip_scores_and_embeddings(src, dest, batch_size=512):
    dataset = wds.DataPipeline(
        wds.SimpleShardList(str(src)),
        wds.split_by_worker,
        wds.tarfile_to_samples(),
        wds.rename(
            __key__="__key__",
            dino_image="jpg",
            image="jpg",
            street_clip="street_clip.npy",
            json="json",
        ),
        wds.decode(
            ImageHandler("pilrgb", ["dino_image"])
        ),  # avoid encoding decoding jpeg for true
        wds.map_dict(
            dino_image=augmentation_dinov2,
            image=lambda x: x,
            street_clip=lambda x: x,
            json=lambda x: x,
        ),
        wds.to_tuple(
            "__key__",
            "dino_image",
            "street_clip",
            "image",
            "json",
        ),
        wds.batched(batch_size),
    )
    loader = wds.WebLoader(dataset, num_workers=8, batch_size=None)
    with wds.TarWriter(str(dest)) as sink:
        for batch in tqdm(loader, total=10000 // batch_size):
            (
                keys,
                dino_image,
                street_clip,
                image,
                json,
            ) = batch
            dino_image = dino_image.to(device)
            with torch.no_grad():
                dino_embedding = dinov2_model(dino_image).cpu().numpy()
            for i in range(len(keys)):
                sample = {
                    "__key__": keys[i],
                    "jpg": image[i],
                    "street_clip.npy": street_clip[i],
                    "json": json[i],
                    "dinov2_vitl14_registers.npy": dino_embedding[i],
                }
                sink.write(sample)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", help="path to source files")
    parser.add_argument("--dest", help="path to destination files")
    parser.add_argument("--shard_id", help="shard id")
    args = parser.parse_args()

    src = Path(args.src)
    list_of_shards = list(src.glob("*.tar"))
    list_of_shards.sort()
    shard = str(list_of_shards[int(args.shard_id)]).split("/")[-1]
    dest = Path(args.dest)
    dest.mkdir(exist_ok=True, parents=True)
    batch_size = 256

    print(f"Loading {shard}")

    tar_name = shard.split(".")[0]

    src_shard = src / shard  # f"{{{tar_name}...{tar_name}}}.tar"

    print(f"Processing {src_shard} to {dest / shard}")
    add_clip_scores_and_embeddings(src_shard, dest / shard, batch_size)
