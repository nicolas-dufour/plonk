import numpy as np
import pandas as pd
import torch
import random
import pickle
from os.path import join
from os.path import isfile
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision.transforms import (
    Compose,
    RandomCrop,
    CenterCrop,
    RandomHorizontalFlip,
    ToTensor,
)
import time
from torchvision.transforms import GaussianBlur
from torchvision import transforms
from pathlib import Path
import json
from tqdm import tqdm
import multiprocessing as mp
import ctypes


def normalize(lat, lon):
    """Used to put all lat lon inside ±90 and ±180."""
    lat = (lat + 90) % 360 - 90
    if lat > 90:
        lat = 180 - lat
        lon += 180
    lon = (lon + 180) % 360 - 180
    return lat, lon


def collate_fn(batch):
    """Collate function for the dataloader.
    Args:
        batch (list): list of dictionaries with keys "img", "gps", "idx" and optionally "label"
    Returns:
        dict: dictionary with keys "img", "gps", "idx" and optionally "label"
    """
    keys = list(batch[0].keys())
    if "weight" in batch[0].keys():
        keys.remove("weight")
    output = {}
    for key in [
        "idx",
        "unique_country",
        "unique_region",
        "unique_sub-region",
        "unique_city",
        "img_idx",
        "text",
    ]:
        if key in keys:
            idx = [x[key] for x in batch]
            output[key] = idx
            keys.remove(key)
    if "img" in keys and isinstance(batch[0]["img"], Image.Image):
        output["img"] = [x["img"] for x in batch]
        keys.remove("img")
    for key in keys:
        if not ("text" in key):
            output[key] = torch.stack([x[key] for x in batch])
    return output


def collate_fn_streetclip(batch):
    """Collate function for the dataloader.
    Args:
        batch (list): list of dictionaries with keys "img", "gps", "idx" and optionally "label"
    Returns:
        dict: dictionary with keys "img", "gps", "idx" and optionally "label"
    """
    keys = list(batch[0].keys())
    if "weight" in batch[0].keys():
        keys.remove("weight")
    output = {}
    for key in [
        "idx",
        "unique_country",
        "unique_region",
        "unique_sub-region",
        "unique_city",
        "img_idx",
        "img",
        "text",
    ]:
        if key in keys:
            idx = [x[key] for x in batch]
            output[key] = idx
            keys.remove(key)
    for key in keys:
        if not ("text" in key):
            output[key] = torch.stack([x[key] for x in batch])
    return output


def collate_fn_denstity(batch):
    """Collate function for the dataloader.
    Args:
        batch (list): list of dictionaries with keys "img", "gps", "idx" and optionally "label"
    Returns:
        dict: dictionary with keys "img", "gps", "idx" and optionally "label"
    """
    keys = list(batch[0].keys())
    if "weight" in batch[0].keys():
        keys.remove("weight")
    # Sample indices based on the weights
    weights = np.array([x["weight"] for x in batch])
    normalized_weights = weights / np.sum(weights)
    sampled_indices = np.random.choice(
        len(batch), size=len(batch), p=normalized_weights, replace=True
    )
    output = {}
    for key in [
        "idx",
        "unique_country",
        "unique_region",
        "unique_sub-region",
        "unique_city",
        "img_idx",
        "text",
    ]:
        if key in keys:
            idx = [batch[i][key] for i in sampled_indices]
            output[key] = idx
            keys.remove(key)
    for key in keys:
        if not ("text" in key):
            output[key] = torch.stack([batch[i][key] for i in sampled_indices])
    return output


def collate_fn_streetclip_denstity(batch):
    """Collate function for the dataloader.
    Args:
        batch (list): list of dictionaries with keys "img", "gps", "idx" and optionally "label"
    Returns:
        dict: dictionary with keys "img", "gps", "idx" and optionally "label"
    """
    keys = list(batch[0].keys())
    if "weight" in batch[0].keys():
        keys.remove("weight")
    # Sample indices based on the weights
    weights = np.array([x["weight"] for x in batch])
    normalized_weights = weights / np.sum(weights)
    sampled_indices = np.random.choice(
        len(batch), size=len(batch), p=normalized_weights, replace=True
    )
    output = {}
    for key in [
        "idx",
        "unique_country",
        "unique_region",
        "unique_sub-region",
        "unique_city",
        "img_idx",
        "img",
        "text",
    ]:
        if key in keys:
            idx = [batch[i][key] for i in sampled_indices]
            output[key] = idx
            keys.remove(key)
    for key in keys:
        if not ("text" in key):
            output[key] = torch.stack([batch[i][key] for i in sampled_indices])
    return output


def collate_fn_contrastive(batch):
    """Collate function for the dataloader.
    Args:
        batch (list): list of dictionaries with keys "img", "gps", "idx" and optionally "label"
    Returns:
        dict: dictionary with keys "img", "gps", "idx" and optionally "label"
    """
    output = collate_fn(batch)
    pos_img = torch.stack([x["pos_img"] for x in batch])
    output["pos_img"] = pos_img
    return output


def collate_fn_contrastive_density(batch):
    """Collate function for the dataloader.
    Args:
        batch (list): list of dictionaries with keys "img", "gps", "idx" and optionally "label"
    Returns:
        dict: dictionary with keys "img", "gps", "idx" and optionally "label"
    """
    keys = list(batch[0].keys())
    if "weight" in batch[0].keys():
        keys.remove("weight")
    # Sample indices based on the weights
    weights = np.array([x["weight"] for x in batch])
    normalized_weights = weights / np.sum(weights)
    sampled_indices = np.random.choice(
        len(batch), size=len(batch), p=normalized_weights, replace=True
    )
    output = {}
    for key in [
        "idx",
        "unique_country",
        "unique_region",
        "unique_sub-region",
        "unique_city",
        "img_idx",
    ]:
        if key in keys:
            idx = [batch[i][key] for i in sampled_indices]
            output[key] = idx
            keys.remove(key)
    for key in keys:
        if not ("text" in key):
            output[key] = torch.stack([batch[i][key] for i in sampled_indices])
    return output


class iNaturalist(Dataset):
    def __init__(
        self,
        path,
        transforms,
        split="train",
        output_type="image",
        embedding_name="dinov2",
    ):
        super().__init__()
        self.split = split
        with open(Path(path) / f"{split}.json", "r") as f:
            self.metadata = json.load(f)
        self.metadata = [
            datapoint
            for datapoint in self.metadata["images"]
            if "latitude" in datapoint and datapoint["latitude"] is not None
        ]
        self.path = path
        self.transforms = transforms
        self.output_type = output_type
        self.embedding_name = embedding_name

        self.collate_fn = collate_fn

    def __getitem__(self, i):
        output = {}
        if "image" in self.output_type:
            image_path = Path(self.path) / "images" / self.metadata[i]["file_name"]
            img = self.transforms(Image.open(image_path))
            output["img"] = img
        if "emb" in self.output_type:
            emb_path = (
                Path(self.path)
                / "embeddings"
                / self.embedding_name
                / self.metadata[i]["file_name"].replace(".jpg", ".npy")
            )
            output["emb"] = torch.tensor(np.load(emb_path))
        lat, lon = normalize(
            self.metadata[i]["latitude"], self.metadata[i]["longitude"]
        )
        output["gps"] = torch.tensor(
            [np.radians(lat), np.radians(lon)], dtype=torch.float
        )
        output["idx"] = i
        output["img_idx"] = self.metadata[i]["id"]
        return output

    def __len__(self):
        return len(self.metadata)


class OSV5M(Dataset):
    csv_dtype = {"category": str, "country": str, "city": str}  # Don't remove.

    def __init__(
        self,
        path,
        transforms,
        split="train",
        class_name=None,
        aux_data=[],
        is_baseline=False,
        areas=["country", "region", "sub-region", "city"],
        streetclip=False,
        suff="",
        blur=False,
        output_type="image",
        embedding_name="dinov2",
    ):
        """Initializes the dataset.
        Args:
            path (str): path to the dataset
            transforms (torchvision.transforms): transforms to apply to the images
            split (str): split to use (train, val, test)
            class_name (str): category to use (e.g. "city")
            aux_data (list of str): auxilliary datas to use
            areas (list of str): regions to perform accuracy
            streetclip (bool): if the model is streetclip, do not use transform
            suff (str): suffix of test csv
            blur (bool): blur bottom of images or not
            output_type (str): type of output (image or emb)
        """
        self.suff = suff
        self.path = path
        self.aux = len(aux_data) > 0
        self.aux_list = aux_data
        self.split = split
        if split == "select":
            self.df = self.load_split(split)
            split = "test"
        else:
            self.df = self.load_split(split)
        self.split = split
        if "image" in output_type:
            self.image_data_folder = join(
                path,
                "images",
                ("train" if split == "val" else split),
            )
            self.image_dict_names = {}
            for root, _, files in os.walk(self.image_data_folder):
                for file in files:
                    self.image_dict_names[file] = os.path.join(root, file)
        if "emb" in output_type:
            self.emb_data_folder = join(
                path,
                "embeddings",
                embedding_name,
                ("train" if split == "val" else split),
            )
            self.emb_dict_names = {}
            for root, _, files in os.walk(self.emb_data_folder):
                for file in files:
                    self.emb_dict_names[file] = os.path.join(root, file)

        self.output_type = output_type

        self.is_baseline = is_baseline
        if self.aux:
            self.aux_data = {}
            for col in self.aux_list:
                if col in ["land_cover", "climate", "soil"]:
                    self.aux_data[col] = pd.get_dummies(self.df[col], dtype=float)
                    if col == "climate":
                        for i in range(31):
                            if not (i in list(self.aux_data[col].columns)):
                                self.aux_data[col][i] = 0
                        desired_order = [i for i in range(31)]
                        desired_order.remove(20)
                        self.aux_data[col] = self.aux_data[col][desired_order]
                else:
                    self.aux_data[col] = self.df[col].apply(lambda x: [x])

        self.areas = ["_".join(["unique", area]) for area in areas]
        if class_name is None:
            self.class_name = class_name
        elif "quadtree" in class_name:
            self.class_name = class_name
        else:
            self.class_name = "_".join(["unique", class_name])
        ex = self.extract_classes(self.class_name)
        self.df = self.df[
            ["id", "latitude", "longitude", "weight"] + self.areas + ex
        ].fillna("NaN")
        if self.class_name in self.areas:
            self.df.columns = list(self.df.columns)[:-1] + [self.class_name + "_2"]
        self.transforms = transforms
        self.collate_fn = collate_fn
        self.collate_fn_density = collate_fn_denstity
        self.blur = blur
        self.streetclip = streetclip
        if self.streetclip:
            self.collate_fn = collate_fn_streetclip
            self.collate_fn_density = collate_fn_streetclip_denstity

    def load_split(self, split):
        """Returns a new dataset with the given split."""
        start_time = time.time()
        if split == "test":
            df = pd.read_csv(join(self.path, "test.csv"), dtype=self.csv_dtype)
            # extract coord
            longitude = df["longitude"].values
            latitude = df["latitude"].values
            # Create bins
            num_bins = 100
            lon_bins = np.linspace(longitude.min(), longitude.max(), num_bins)
            lat_bins = np.linspace(latitude.min(), latitude.max(), num_bins)
            # compute density and weights
            hist, _, _ = np.histogram2d(longitude, latitude, bins=[lon_bins, lat_bins])
            weights = 1.0 / np.power(hist[df["lon_bin"], df["lat_bin"]], 0.75)
            normalized_weights = weights / np.sum(weights)
            df["weight"] = normalized_weights
            return df
        elif split == "select":
            df = pd.read_csv(join(self.path, "select.csv"), dtype=self.csv_dtype)
            # extract coord
            longitude = df["longitude"].values
            latitude = df["latitude"].values
            # Create bins
            num_bins = 100
            lon_bins = np.linspace(longitude.min(), longitude.max(), num_bins)
            lat_bins = np.linspace(latitude.min(), latitude.max(), num_bins)
            # compute density and weights
            hist, _, _ = np.histogram2d(longitude, latitude, bins=[lon_bins, lat_bins])
            weights = 1.0 / np.power(hist[df["lon_bin"], df["lat_bin"]], 0.75)
            normalized_weights = weights / np.sum(weights)
            df["weight"] = normalized_weights
            return df
        else:
            if len(self.suff) == 0:
                df = pd.read_csv(join(self.path, "train.csv"), dtype=self.csv_dtype)
            else:
                df = pd.read_csv(
                    join(self.path, "train" + "_" + self.suff + ".csv"),
                    dtype=self.csv_dtype,
                )

        # extract coord
        longitude = df["longitude"].values
        latitude = df["latitude"].values
        # Create bins
        num_bins = 100
        lon_bins = np.linspace(longitude.min(), longitude.max(), num_bins)
        lat_bins = np.linspace(latitude.min(), latitude.max(), num_bins)
        # compute density and weights
        hist, _, _ = np.histogram2d(longitude, latitude, bins=[lon_bins, lat_bins])
        weights = 1.0 / np.power(hist[df["lon_bin"], df["lat_bin"]], 0.75)
        normalized_weights = weights / np.sum(weights)
        df["weight"] = normalized_weights

        test_df = df.sample(
            n=int(0.1 * len(df)),
            weights=normalized_weights,
            replace=False,
            random_state=42,
        )

        end_time = time.time()
        print(f"Loading {split} dataset took {(end_time - start_time):.2f} seconds")

        if split == "val":
            return test_df
        else:
            return df.drop(test_df.index)

    def extract_classes(self, tag=None):
        """Extracts the categories from the dataset."""
        if tag is None:
            self.has_labels = False
            return []
        splits = ["train", "test"] if self.is_baseline else ["train"]
        # splits = ["train", "test"]
        print(f"Loading categories from {splits}")

        # concatenate all categories from relevant splits to find the unique ones.
        self.categories = sorted(
            pd.concat(
                [pd.read_csv(join(self.path, f"{split}.csv"))[tag] for split in splits]
            )
            .fillna("NaN")
            .unique()
            .tolist()
        )

        if "NaN" in self.categories:
            self.categories.remove("NaN")
            if self.split != "test":
                self.df = self.df.dropna(subset=[tag])
        # compute the total number of categories - this name is fixed and will be used as a lookup during init
        self.num_classes = len(self.categories)

        # create a mapping from category to index
        self.category_to_index = {
            category: i for i, category in enumerate(self.categories)
        }
        self.has_labels = True
        return [tag]

    def __getitem__(self, i):
        """Returns an item from the dataset.
        Args:
            i (int): index of the item
        Returns:
            dict: dictionary with keys "img", "gps", "idx" and optionally "label"
        """
        x = list(self.df.iloc[i])  # id, latitude, longitude, {category}
        output = {}
        if "image" in self.output_type:
            if self.streetclip:
                img = Image.open(self.image_dict_names[f"{int(x[0])}.jpg"])
            elif self.blur:
                img = transforms.ToTensor()(
                    Image.open(self.image_dict_names[f"{int(x[0])}.jpg"])
                )
                u = GaussianBlur(kernel_size=13, sigma=2.0)
                bottom_part = img[:, -14:, :].unsqueeze(0)
                blurred_bottom = u(bottom_part)
                img[:, -14:, :] = blurred_bottom.squeeze()
                img = self.transforms(transforms.ToPILImage()(img))
            else:
                img = self.transforms(
                    Image.open(self.image_dict_names[f"{int(x[0])}.jpg"])
                )
            output["img"] = img
        if "emb" in self.output_type:
            output["emb"] = torch.FloatTensor(
                np.load(self.emb_dict_names[f"{int(x[0])}.npy"])
            )

        lat, lon = normalize(x[1], x[2])
        gps = torch.FloatTensor([np.radians(lat), np.radians(lon)]).squeeze(0)

        output.update(
            {
                "gps": gps,
                "idx": i,
                "img_idx": int(x[0]),
                "weight": x[3],
            }
        )

        for count, area in enumerate(self.areas):
            output[area] = x[
                count + 4
            ]  #'country': x[3], 'region': x[4], 'sub-region': x[5], 'city': x[6]}

        if self.has_labels:
            if x[-1] in self.categories:
                output["label"] = torch.LongTensor(
                    [self.category_to_index[x[-1]]]
                ).squeeze(-1)
            else:
                output["label"] = torch.LongTensor([-1]).squeeze(-1)
        if self.aux:
            for col in self.aux_list:
                output[col] = torch.FloatTensor(self.aux_data[col].iloc[i])
        return output

    def __len__(self):
        return len(self.df)


class ContrastiveOSV5M(OSV5M):
    def __init__(
        self,
        path,
        transforms,
        split="train",
        class_name=None,
        aux_data=[],
        class_name2=None,
        blur=False,
    ):
        """
        class_name2 (str): if not None, we do contrastive an other class than the one specified for classif
        """
        super().__init__(
            path,
            transforms,
            split=split,
            class_name=class_name,
            aux_data=aux_data,
            blur=blur,
        )
        self.add_label = False
        if not (class_name2 is None) and split != "test" and split != "select":
            self.add_label = True
            self.class_name = class_name2
            self.extract_classes_contrastive(tag=class_name2)
        self.df = self.df.reset_index(drop=True)
        self.dict_classes = {
            value: indices.tolist()
            for value, indices in self.df.groupby(self.class_name).groups.items()
        }
        self.collate_fn = collate_fn_contrastive
        self.random_crop = RandomCrop(224)  # use when no positive image is available

    def sample_positive(self, i):
        """
        sample positive image from the same city, country if it is available
        otherwise, apply different crop to the image
        """
        x = self.df.iloc[i]  # id, latitude, longitude, {category}
        class_name = x[self.class_name]
        idxs = self.dict_classes[class_name]
        idxs.remove(i)

        if len(idxs) > 0:
            idx = random.choice(idxs)
            x = self.df.iloc[idx]
            pos_img = self.transforms(
                Image.open(self.dict_names[f"{int(x['id'])}.jpg"])
            )
        else:
            pos_img = self.random_crop(
                self.transforms(Image.open(self.dict_names[f"{int(x['id'])}.jpg"]))
            )
        return pos_img

    def extract_classes_contrastive(self, tag=None):
        """Extracts the categories from the dataset."""
        if tag is None:
            self.has_labels = False
            return []
        splits = ["train", "test"] if self.is_baseline else ["train"]
        # splits = ["train", "test"]
        print(f"Loading categories from {splits}")

        # concatenate all categories from relevant splits to find the unique ones.
        categories = sorted(
            pd.concat(
                [pd.read_csv(join(self.path, f"{split}.csv"))[tag] for split in splits]
            )
            .fillna("NaN")
            .unique()
            .tolist()
        )
        # create a mapping from category to index
        self.contrastive_category_to_index = {
            category: i for i, category in enumerate(categories)
        }

    def __getitem__(self, i):
        output = super().__getitem__(i)
        pos_img = self.sample_positive(i)
        output["pos_img"] = pos_img
        if self.add_label:
            output["label_contrastive"] = torch.LongTensor(
                [self.contrastive_category_to_index[self.df[self.class_name].iloc[i]]]
            ).squeeze(-1)
        return output


class TextContrastiveOSV5M(OSV5M):
    def __init__(
        self,
        path,
        transforms,
        split="train",
        class_name=None,
        aux_data=[],
        blur=False,
    ):
        super().__init__(
            path,
            transforms,
            split=split,
            class_name=class_name,
            aux_data=aux_data,
            blur=blur,
        )
        self.df = self.df.reset_index(drop=True)

    def get_text(self, i):
        """
        sample positive image from the same city, country if it is available
        otherwise, apply different crop to the image
        """
        x = self.df.iloc[i]  # id, latitude, longitude, {category}
        l = [
            name.split("_")[-1]
            for name in [
                x["unique_city"],
                x["unique_sub-region"],
                x["unique_region"],
                x["unique_country"],
            ]
        ]

        pre = False
        sentence = "An image of "
        if l[0] != "NaN":
            sentence += "the city of "
            sentence += l[0]
            pre = True

        if l[1] != "NaN":
            if pre:
                sentence += ", in "
            sentence += "the area of "
            sentence += l[1]
            pre = True

        if l[2] != "NaN":
            if pre:
                sentence += ", in "
            sentence += "the region of "
            sentence += l[2]
            pre = True

        if l[3] != "NaN":
            if pre:
                sentence += ", in "
            sentence += l[3]

        return sentence

    def __getitem__(self, i):
        output = super().__getitem__(i)
        output["text"] = self.get_text(i)
        return output


import os
import json


class Baseline(Dataset):
    def __init__(
        self,
        path,
        which,
        transforms,
    ):
        """Initializes the dataset.
        Args:
            path (str): path to the dataset
            which (str): which baseline to use (im2gps, im2gps3k)
            transforms (torchvision.transforms): transforms to apply to the images
        """
        baselines = {
            "im2gps": self.load_im2gps,
            "im2gps3k": self.load_im2gps,
            "yfcc4k": self.load_yfcc4k,
        }
        self.path = path
        self.samples = baselines[which]()
        self.transforms = transforms
        self.collate_fn = collate_fn
        self.class_name = which

    def load_im2gps(
        self,
    ):
        json_path = join(self.path, "info.json")
        with open(json_path) as f:
            data = json.load(f)

        samples = []
        for f in os.listdir(join(self.path, "images")):
            if len(data[f]):
                lat = float(data[f][-4].replace("latitude: ", ""))
                lon = float(data[f][-3].replace("longitude: ", ""))
                samples.append((f, lat, lon))

        return samples

    def load_yfcc4k(
        self,
    ):
        samples = []
        with open(join(self.path, "info.txt")) as f:
            lines = f.readlines()
        for line in lines:
            x = line.split("\t")
            f, lon, lat = x[1], x[12], x[13]
            samples.append((f + ".jpg", float(lat), float(lon)))

        return samples

    def __getitem__(self, i):
        """Returns an item from the dataset.
        Args:
            i (int): index of the item
        Returns:
            dict: dictionary with keys "img", "gps", "idx" and optionally "label"
        """
        img_path, lat, lon = self.samples[i]
        img = self.transforms(
            Image.open(join(self.path, "images", img_path)).convert("RGB")
        )
        lat, lon = normalize(lat, lon)
        gps = torch.FloatTensor([np.radians(lat), np.radians(lon)]).squeeze(0)

        return {
            "img": img,
            "gps": gps,
            "idx": i,
        }

    def __len__(self):
        return len(self.samples)


null_transform = lambda x: x
