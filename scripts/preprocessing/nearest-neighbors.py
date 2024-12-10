import sys, os
import json
from PIL import Image
from tqdm import tqdm
from os.path import dirname, join

sys.path.append(dirname(dirname(__file__)))

import torch
from transformers import AutoImageProcessor, AutoModel
from transformers import CLIPProcessor, CLIPModel
from transformers import pipeline

from data.data import osv5m
from json_stream import streamable_list

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_clip():
    model = CLIPModel.from_pretrained("laion/CLIP-ViT-L-14-laion2B-s32B-b82K")
    processor = CLIPProcessor.from_pretrained("laion/CLIP-ViT-L-14-laion2B-s32B-b82K")
    return processor, model.to(DEVICE)


def load_model_dino():
    model = AutoModel.from_pretrained("facebook/dinov2-base")
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    return processor, model.to(DEVICE)


def compute_dino(processor, model, x):
    inputs = processor(images=x[0], return_tensors="pt", device=DEVICE).to(DEVICE)
    outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state.cpu().numpy()
    for i in range(len(x[0])):
        yield [last_hidden_states[i].tolist(), x[1][i], x[2][i], x[3][i]]


def compute_clip(processor, model, x):
    inputs = processor(images=x[0], return_tensors="pt", device=DEVICE).to(DEVICE)
    features = model.get_image_features(**inputs)
    features /= features.norm(dim=-1, keepdim=True)
    features = features.cpu().numpy()
    for i in range(len(x[0])):
        yield [features[i].tolist(), x[1][i], x[2][i], x[3][i]]


def get_batch(dataset, batch_size):
    data, lats, lons, ids = [], [], [], []
    for i in range(len(dataset)):
        id, lat, lon = dataset.df.iloc[i]
        data.append(Image.open(join(dataset.image_folder, f"{int(id)}.jpg")))
        lats.append(lat)
        lons.append(lon)
        ids.append(id)
        if len(data) == batch_size:
            yield data, lats, lons, ids
            data, lats, lons, ids = [], [], [], []

    if len(data) > 0:
        yield data, lats, lons, ids
        data, lats, lons, ids = [], [], [], []


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--compute_features", action="store_true")
    parser.add_argument("--compute_nearest", action="store_true")
    parser.add_argument("--json_path", default="features")
    parser.add_argument("--which", type=str, default="clip", choices=["clip", "dino"])
    args = parser.parse_args()
    json_path = join(args.json_path, args.which)

    os.makedirs(json_path, exist_ok=True)
    if args.compute_features:
        processor, model = (
            load_model_clip() if args.which == "clip" else load_model_dino()
        )
        compute_fn = compute_clip if args.which == "clip" else compute_dino

        for split in ["test"]:  #'train',
            # open existing json and read as dictionary
            json_path_ = join(json_path, f"{split}.json")

            dataset = OSV5M(
                "datasets/osv5m", transforms=None, split=split, dont_split=True
            )

            @torch.no_grad()
            def compute(batch_size):
                for data in tqdm(
                    get_batch(dataset, batch_size),
                    total=len(dataset) // batch_size,
                    desc=f"Computing {split} on {args.which}",
                ):
                    features = compute_fn(processor, model, data)
                    for feature, lat, lon, id in features:
                        yield feature, lat, lon, id

            data = streamable_list(compute(args.batch_size))
            json.dump(data, open(json_path_, "w"), indent=4)

    if args.compute_nearest:
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np

        train, test = [
            json.load(open(join(json_path, f"{split}.json"), "r"))
            for split in ["train", "test"]
        ]

        def get_neighbors(k=10):
            for i, test_data in enumerate(tqdm(test)):
                feature, lat, lon, id = test_data
                features_train = np.stack(
                    [np.array(train_data[0]) for train_data in train]
                )
                cs = np.squeeze(
                    cosine_similarity(np.expand_dims(feature, axis=0), features_train),
                    axis=0,
                )
                i = np.argsort(cs)[-k:][::-1].tolist()
                yield [
                    {n: x}
                    for idx in i
                    for n, x in zip(
                        ["feature", "lat", "lon", "id", "distance"],
                        train[idx]
                        + [
                            cs[idx],
                        ],
                    )
                ]

        data = streamable_list(get_neighbors())
        json.dump(data, open(join(json_path, "nearest.json"), "w"), indent=4)
