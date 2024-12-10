import os
import sys
import PIL
import json
import torch
import numpy as np
import pandas as pd
import operator

from PIL import Image
from itertools import cycle
from tqdm.auto import tqdm, trange
from os.path import join
from PIL import Image

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F

from backbone import get_backbone
from utils import haversine, get_filenames, get_match_values, compute_print_accuracy


def compute_features(path, data_dir, csv_file, tag, args):
    data = GeoDataset(data_dir, csv_file, tag=tag)
    if not os.path.isdir(test_features_dir) or len(
        os.listdir(test_features_dir)
    ) != len(data):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, transform, inference, collate_fn = get_backbone(args.name)
        dataloader = DataLoader(
            data,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=8,
            collate_fn=collate_fn,
        )
        model = model.to(device)
        os.makedirs(path, exist_ok=True)

        for i, x in enumerate(tqdm(dataloader)):
            image_ids, features = inference(model, x)
            # save features as numpy array
            for j, image_id in zip(range(features.shape[0]), image_ids):
                np.save(join(path, f"{image_id}.npy"), features[j].unsqueeze(0).numpy())


def get_results(args, train_test):
    import joblib

    if not os.path.isfile(join(args.features_parent, ".cache", "1-nn.pkl")):
        import faiss, glob, bisect

        # import sys; sys.exit(0)
        indexes = [
            get_filenames(idx) for idx in tqdm(range(1, 6), desc="Loading indexes...")
        ]

        train_gt = pd.read_csv(
            join(args.data_parent, args.annotation_file), dtype={"image_id": str}
        )[["image_id", "latitude", "longitude"]]
        test_gt = pd.read_csv(test_path_csv, dtype={"id": str})[
            ["id", "latitude", "longitude"]
        ]

        # make a map between image_id and lat/lon
        train_gt = {
            g[1]["image_id"]: np.array([g[1]["latitude"], g[1]["longitude"]])
            for g in tqdm(
                train_gt.iterrows(), total=len(train_gt), desc="Loading train_gt"
            )
        }
        test_gt = {
            g[1]["id"]: np.array([g[1]["latitude"], g[1]["longitude"]])
            for g in tqdm(
                test_gt.iterrows(), total=len(test_gt), desc="Loading test_gt"
            )
        }

        train_test = []
        os.makedirs(join(args.features_parent, ".cache"), exist_ok=True)
        for f in tqdm(os.listdir(test_features_dir)):
            query_vector = np.load(join(test_features_dir, f))

            neighbors = []
            for index, ids in indexes:
                distances, indices = index.search(query_vector, 1)
                distances, indices = np.squeeze(distances), np.squeeze(indices)
                bisect.insort(
                    neighbors, (ids[indices], distances), key=operator.itemgetter(1)
                )

            neighbors = list(reversed(neighbors))
            train_gps = train_gt[neighbors[0][0].replace(".npy", "")][None, :]
            test_gps = test_gt[f.replace(".npy", "")][None, :]
            train_test.append((train_gps, test_gps))
        joblib.dump(train_test, join(args.features_parent, ".cache", "1-nn.pkl"))
    else:
        train_test = joblib.load(join(args.features_parent, ".cache", "1-nn.pkl"))

    return train_test


if __name__ == "__main__":
    # make a train/eval argparser
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=int, default=1)  # maybe need to remove/refactor
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument(
        "--annotation_file", type=str, required=False, default="train.csv"
    )
    parser.add_argument("--name", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--features_parent", type=str, default="faiss/")
    parser.add_argument("--data_parent", type=str, default="data/")
    parser.add_argument("--test", action="store_true")

    args = parser.parse_args()
    args.features_parent = join(args.features_parent, args.name)
    if args.test:
        csv_file = join(args.data_parent, "test.csv")
        data_dir = join(args.data_parent, "test")
        path = join(args.features_parent, "features-test")
        model = get_backbone(args.name)
        compute_features(path, data_dir, csv_file, tag="id", args=args)
        train_test = get_results(args, train_test)

        from collections import Counter

        N, pos = Counter(), Counter()
        for train_gps, test_gps in tqdm(train_test, desc="Computing accuracy..."):
            get_match_values(train_gps, test_gps, N, pos)

        for train_gps, test_gps in tqdm(train_test, desc="Computing haversine..."):
            haversine(train_gps, test_gps, N, pos)

        compute_print_accuracy(N, pos)
    else:
        csv_file = join(args.data_parent, args.annotation_file)
        path = join(args.features_parent, f"features-{args.id}")
        data_dir = join(args.data_parent, f"images-{args.id}", "train")
        compute_features(path, data_dir, csv_file, tag="image_id", args=args)
