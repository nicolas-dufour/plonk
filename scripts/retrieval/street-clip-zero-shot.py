import traceback
import os
import sys
import PIL
import json
import torch
import numpy as np
import pandas as pd
import operator
import joblib
import reverse_geocoder

from PIL import Image
from itertools import cycle
from tqdm.auto import tqdm, trange
from os.path import join
from PIL import Image

from tqdm import tqdm
from collections import Counter
from transformers import CLIPProcessor, CLIPModel
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from utils import haversine


class GeoDataset(Dataset):
    def __init__(self, image_folder, annotation_file, tag="image_id"):
        self.image_folder = image_folder
        gt = pd.read_csv(annotation_file, dtype={tag: str})
        files = set([f.replace(".jpg", "") for f in os.listdir(image_folder)])
        gt = gt[gt[tag].isin(files)]
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
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
        pil = PIL.Image.open(fp)
        proc = self.processor(images=pil, return_tensors="pt")
        proc["image_id"] = self.gt[idx][0]
        return proc


@torch.no_grad()
def compute_features_clip(img, model):
    image_ids = img.data.pop("image_id")
    image_input = img.to(model.device)
    image_input["pixel_values"] = image_input["pixel_values"].squeeze(1)
    features = model.get_image_features(**image_input)
    features /= features.norm(dim=-1, keepdim=True)
    return image_ids, features.cpu()


def get_prompts(country, region, sub_region, city):
    a = country if country != "" else None
    b, c, d = None, None, None
    if a is not None:
        b = country + ", " + region if region != "" else None
        if b is not None:
            c = (
                country + ", " + region + ", " + sub_region
                if sub_region != ""
                else None
            )
            d = (
                country + ", " + region + ", " + sub_region + ", " + city
                if city != ""
                else None
            )
    return a, b, c, d


if __name__ == "__main__":
    # make a train/eval argparser
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--annotation_file", type=str, required=False, default="train.csv"
    )
    parser.add_argument(
        "--features_parent", type=str, default="/home/isig/gaia-v2/faiss/street-clip"
    )
    parser.add_argument(
        "--data_parent", type=str, default="/home/isig/gaia-v2/loic-data/"
    )

    args = parser.parse_args()
    test_path_csv = join(args.data_parent, "test.csv")
    test_image_dir = join(args.data_parent, "test")
    save_path = join(args.features_parent, "indexes/test.index")
    test_features_dir = join(args.features_parent, "indexes/features-test")

    processor = CLIPProcessor.from_pretrained("geolocal/StreetCLIP")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained("geolocal/StreetCLIP").to(device)

    @torch.no_grad()
    def compute_text_features_clip(text):
        text_pt = processor(text=text, return_tensors="pt").to(device)
        features = model.get_text_features(**text_pt)
        features /= features.norm(dim=-1, keepdim=True)
        return features.cpu().squeeze(0).numpy()

    import country_converter as coco

    if not os.path.isfile("text_street-clip-features.pkl"):
        if not os.path.isfile("rg_cities1000.csv"):
            os.system(
                "wget https://raw.githubusercontent.com/thampiman/reverse-geocoder/master/reverse_geocoder/rg_cities1000.csv"
            )

        cities = pd.read_csv("rg_cities1000.csv")
        cities = cities[["lat", "lon", "name", "admin1", "admin2", "cc"]]
        reprs = {0: {}, 1: {}, 2: {}, 3: {}}
        for line in tqdm(
            cities.iterrows(), total=len(cities), desc="Creating hierarchy"
        ):
            lat, lon, city, region, sub_region, cc = line[1]
            try:
                city, region, sub_region, cc = [
                    ("" if pd.isna(x) else x)
                    for x in [
                        city,
                        region,
                        sub_region,
                        coco.convert(cc, to="name_short"),
                    ]
                ]
                a, b, c, d = get_prompts(cc, region, sub_region, city)
                if a is not None:
                    if a not in reprs[0]:
                        reprs[0][a] = {
                            "gps": {(lat, lon)},
                            "embedding": compute_text_features_clip(a),
                        }
                    else:
                        reprs[0][a]["gps"].add((lat, lon))

                if b is not None:
                    if b not in reprs[1]:
                        reprs[1][b] = {
                            "gps": {(lat, lon)},
                            "embedding": compute_text_features_clip(b),
                        }
                    else:
                        reprs[1][b]["gps"].add((lat, lon))

                if c is not None:
                    if c not in reprs[2]:
                        reprs[2][c] = {
                            "gps": {(lat, lon)},
                            "embedding": compute_text_features_clip(c),
                        }
                    else:
                        reprs[2][c]["gps"].add((lat, lon))

                if d is not None:
                    if d not in reprs[3]:
                        reprs[3][d] = {
                            "gps": {(lat, lon)},
                            "embedding": compute_text_features_clip(
                                d.replace(", , ", ", ")
                            ),
                        }
                    else:
                        reprs[3][d]["gps"].add((lat, lon))
            except Exception as e:
                # print stack trace into file log.txt
                with open("log.txt", "a") as f:
                    print(traceback.format_exc(), file=f)

        reprs[-1] = {"": {"gps": (0, 0), "embedding": compute_text_features_clip("")}}

        # compute mean for gps of all 'a' and 'b' and 'c' and 'd'
        for i in range(4):
            for k in reprs[i].keys():
                reprs[i][k]["gps"] = tuple(
                    np.array(list(reprs[i][k]["gps"])).mean(axis=0).tolist()
                )

        joblib.dump(reprs, "text_street-clip-features.pkl")
    else:
        reprs = joblib.load("text_street-clip-features.pkl")

    def get_loc(x):
        location = reverse_geocoder.search(x[0].tolist())[0]
        country = coco.convert(names=location["cc"], to="name_short")
        region = location.get("admin1", "")
        sub_region = location.get("admin2", "")
        city = location.get("name", "")
        a, b, c, d = get_prompts(country, region, sub_region, city)
        return a, b, c, d

    def matches(embed, repr, control, gt, sw=None):
        first_max = max(
            (
                (k, embed.dot(v["embedding"]))
                for k, v in repr.items()
                if sw is None or k.startswith(sw)
            ),
            key=operator.itemgetter(1),
        )
        if first_max[1] > embed.dot(control["embedding"]):
            return repr[first_max[0]]["gps"], gt == first_max[0]
        else:
            return control["gps"], False

    def get_match_values(gt, embed, N, pos):
        xa, xb, xc, xd = get_loc(gt)

        if xa is not None:
            N["country"] += 1
            gps, flag = matches(embed, reprs[0], reprs[-1][""], xa)
            if flag:
                pos["country"] += 1
                if xb is not None:
                    N["region"] += 1
                    gps, flag = matches(embed, reprs[1], reprs[0][xa], xb, sw=xa)
                    if flag:
                        pos["region"] += 1
                        if xc is not None:
                            N["sub-region"] += 1
                            gps, flag = matches(
                                embed, reprs[2], reprs[1][xb], xc, sw=xb
                            )
                            if flag:
                                pos["sub-region"] += 1
                                if xd is not None:
                                    N["city"] += 1
                                    gps, flag = matches(
                                        embed, reprs[3], reprs[2][xc], xd, sw=xc
                                    )
                                    if flag:
                                        pos["city"] += 1
                        else:
                            if xd is not None:
                                N["city"] += 1
                                gps, flag = matches(
                                    embed, reprs[3], reprs[1][xb], xd, sw=xb + ", "
                                )
                                if flag:
                                    pos["city"] += 1

        haversine(np.array(gps)[None, :], np.array(gt), N, pos)

    def compute_print_accuracy(N, pos):
        for k in N.keys():
            pos[k] /= N[k]

        # pretty-print accuracy in percentage with 2 floating points
        print(
            f'Accuracy: {pos["country"]*100.0:.2f} (country), {pos["region"]*100.0:.2f} (region), {pos["sub-region"]*100.0:.2f} (sub-region), {pos["city"]*100.0:.2f} (city)'
        )
        print(
            f'Haversine: {pos["haversine"]:.2f} (haversine), {pos["geoguessr"]:.2f} (geoguessr)'
        )

    import joblib

    data = GeoDataset(test_image_dir, test_path_csv, tag="id")
    test_gt = pd.read_csv(test_path_csv, dtype={"id": str})[
        ["id", "latitude", "longitude"]
    ]
    test_gt = {
        g[1]["id"]: np.array([g[1]["latitude"], g[1]["longitude"]])
        for g in tqdm(test_gt.iterrows(), total=len(test_gt), desc="Loading test_gt")
    }

    with open("/home/isig/gaia-v2/loic/plonk/test3_indices.txt", "r") as f:
        # read lines
        lines = f.readlines()
        # remove whitespace characters like `\n` at the end of each line
        lines = [l.strip() for l in lines]
        # and convert to set
        lines = set(lines)

    train_test = []
    N, pos = Counter(), Counter()
    for f in tqdm(os.listdir(test_features_dir)):
        if f.replace(".npy", "") not in lines:
            continue
        query_vector = np.squeeze(np.load(join(test_features_dir, f)))
        test_gps = test_gt[f.replace(".npy", "")][None, :]
        get_match_values(test_gps, query_vector, N, pos)

    compute_print_accuracy(N, pos)
