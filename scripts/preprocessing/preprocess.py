import pandas as pd
import torch
import numpy as np
from os.path import join
import matplotlib.pyplot as plt
import hydra


class QuadTree(object):
    def __init__(self, data, mins=None, maxs=None, id="", depth=3, do_split=1000):
        self.id = id
        self.data = data

        if mins is None:
            mins = data[["latitude", "longitude"]].to_numpy().min(0)
        if maxs is None:
            maxs = data[["latitude", "longitude"]].to_numpy().max(0)

        self.mins = np.asarray(mins)
        self.maxs = np.asarray(maxs)
        self.sizes = self.maxs - self.mins

        self.children = []

        mids = 0.5 * (self.mins + self.maxs)
        xmin, ymin = self.mins
        xmax, ymax = self.maxs
        xmid, ymid = mids

        if (depth > 0) and (len(self.data) >= do_split):
            # split the data into four quadrants
            data_q1 = data[(data["latitude"] < mids[0]) & (data["longitude"] < mids[1])]
            data_q2 = data[
                (data["latitude"] < mids[0]) & (data["longitude"] >= mids[1])
            ]
            data_q3 = data[
                (data["latitude"] >= mids[0]) & (data["longitude"] < mids[1])
            ]
            data_q4 = data[
                (data["latitude"] >= mids[0]) & (data["longitude"] >= mids[1])
            ]

            # recursively build a quad tree on each quadrant which has data
            if data_q1.shape[0] > 0:
                self.children.append(
                    QuadTree(
                        data_q1,
                        [xmin, ymin],
                        [xmid, ymid],
                        id + "0",
                        depth - 1,
                        do_split=do_split,
                    )
                )
            if data_q2.shape[0] > 0:
                self.children.append(
                    QuadTree(
                        data_q2,
                        [xmin, ymid],
                        [xmid, ymax],
                        id + "1",
                        depth - 1,
                        do_split=do_split,
                    )
                )
            if data_q3.shape[0] > 0:
                self.children.append(
                    QuadTree(
                        data_q3,
                        [xmid, ymin],
                        [xmax, ymid],
                        id + "2",
                        depth - 1,
                        do_split=do_split,
                    )
                )
            if data_q4.shape[0] > 0:
                self.children.append(
                    QuadTree(
                        data_q4,
                        [xmid, ymid],
                        [xmax, ymax],
                        id + "3",
                        depth - 1,
                        do_split=do_split,
                    )
                )

    def unwrap(self):
        if len(self.children) == 0:
            return {self.id: [self.mins, self.maxs, self.data.copy()]}
        else:
            d = dict()
            for child in self.children:
                d.update(child.unwrap())
            return d


def extract(qt, name_new_column):
    cluster = qt.unwrap()
    boundaries, data = {}, []
    id_to_quad = np.array(list(cluster.keys()))
    for i, (id, vs) in zip(np.arange(len(cluster)), cluster.items()):
        (min_lat, min_lon), (max_lat, max_lon), points = vs
        points[name_new_column] = int(i)
        data.append(points)
        boundaries[i] = (
            float(min_lat),
            float(min_lon),
            float(max_lat),
            float(max_lon),
            points["latitude"].mean(),
            points["longitude"].mean(),
        )

    data = pd.concat(data)
    return boundaries, data, id_to_quad


def vizu(name_new_column, df_train, boundaries, save_path):
    plt.hist(df_train[name_new_column], bins=len(boundaries))
    plt.xlabel("Cluster ID")
    plt.ylabel("Number of images")
    plt.title("Cluster distribution")
    plt.yscale("log")
    plt.savefig(join(save_path, f"{name_new_column}_distrib.png"))
    plt.clf()

    plt.scatter(
        df_train["longitude"].to_numpy(),
        df_train["latitude"].to_numpy(),
        c=np.random.permutation(len(boundaries))[df_train[name_new_column].to_numpy()],
        cmap="tab20",
        s=0.1,
        alpha=0.5,
    )
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Quadtree map")
    plt.savefig(join(save_path, f"{name_new_column}_map.png"))


@hydra.main(
    config_path="../../configs/scripts",
    config_name="preprocess",
    version_base=None,
)
def main(cfg):
    data_path = join(cfg.data_dir, "osv5m")
    save_path = cfg.data_dir
    name_new_column = f"quadtree_{cfg.depth}_{cfg.do_split}"

    # Create clusters from train images
    train_fp = join(data_path, f"train.csv")
    df_train = pd.read_csv(train_fp, low_memory=False)

    qt = QuadTree(df_train, depth=cfg.depth, do_split=cfg.do_split)
    boundaries, df_train, id_to_quad = extract(qt, name_new_column)

    vizu(name_new_column, df_train, boundaries, save_path)

    # Save clusters
    boundaries = pd.DataFrame.from_dict(
        boundaries,
        orient="index",
        columns=["min_lat", "min_lon", "max_lat", "max_lon", "mean_lat", "mean_lon"],
    )
    boundaries.to_csv(
        join(save_path, f"{name_new_column}.csv"), index_label="cluster_id"
    )

    # Assign test images to clusters
    test_fp = join(data_path, f"test.csv")
    df_test = pd.read_csv(test_fp)

    above_lat = np.expand_dims(df_test["latitude"].to_numpy(), -1) > np.expand_dims(
        boundaries["min_lat"].to_numpy(), 0
    )
    below_lat = np.expand_dims(df_test["latitude"].to_numpy(), -1) < np.expand_dims(
        boundaries["max_lat"].to_numpy(), 0
    )
    above_lon = np.expand_dims(df_test["longitude"].to_numpy(), -1) > np.expand_dims(
        boundaries["min_lon"].to_numpy(), 0
    )
    below_lon = np.expand_dims(df_test["longitude"].to_numpy(), -1) < np.expand_dims(
        boundaries["max_lon"].to_numpy(), 0
    )

    mask = np.logical_and(
        np.logical_and(above_lat, below_lat), np.logical_and(above_lon, below_lon)
    )

    df_test[name_new_column] = np.argmax(mask, axis=1)

    # save index_to_gps_quadtree file
    lat = torch.tensor(boundaries["mean_lat"])
    lon = torch.tensor(boundaries["mean_lon"])
    coord = torch.stack([lat, lon], dim=-1)
    torch.save(
        coord, join(save_path, f"index_to_gps_quadtree_{cfg.depth}_{cfg.do_split}.pt")
    )

    torch.save(id_to_quad, join(save_path, f"id_to_quad_{cfg.depth}_{cfg.do_split}.pt"))
    # Overwrite test.csv and train.csv
    if cfg.overwrite_csv:
        df_train.to_csv(train_fp, index=False)
        df_test.to_csv(test_fp, index=False)

    df = pd.read_csv(join(data_path, "train.csv"), low_memory=False).fillna("NaN")
    # Compute the average location for each unique country
    country_avg = (
        df.groupby("unique_country")[["latitude", "longitude"]].mean().reset_index()
    )
    country_avg.to_csv(
        join(save_path, "country_center.csv"),
        columns=["unique_country", "latitude", "longitude"],
        index=False,
    )
    # Compute the average location for each unique admin1 (region)
    region_avg = (
        df.groupby(["unique_region"])[["latitude", "longitude"]].mean().reset_index()
    )
    region_avg.to_csv(
        join(save_path, "region_center.csv"),
        columns=["unique_region", "latitude", "longitude"],
        index=False,
    )
    # Compute the average location for each unique admin2 (area)
    area_avg = (
        df.groupby(["unique_sub-region"])[["latitude", "longitude"]]
        .mean()
        .reset_index()
    )
    area_avg.to_csv(
        join(save_path, "sub-region_center.csv"),
        columns=["unique_sub-region", "latitude", "longitude"],
        index=False,
    )
    # Compute the average location for each unique city
    city_avg = (
        df.groupby(["unique_city"])[["latitude", "longitude"]].mean().reset_index()
    )
    city_avg.to_csv(
        join(save_path, "city_center.csv"),
        columns=["unique_city", "latitude", "longitude"],
        index=False,
    )

    for class_name in [
        "unique_country",
        "unique_sub-region",
        "unique_region",
        "unique_city",
    ]:
        # Load CSV data into a Pandas DataFrame
        csv_file = class_name.split("_")[-1] + "_center.csv"
        df = pd.read_csv(join(save_path, csv_file), low_memory=False)

        splits = ["train"]
        categories = sorted(
            pd.concat(
                [
                    pd.read_csv(
                        join(data_path, f"{split}.csv"), low_memory=False
                    )[class_name]
                    for split in splits
                ]
            )
            .fillna("NaN")
            .unique()
            .tolist()
        )

        if "NaN" in categories:
            categories.remove("NaN")

        # compute the total number of categories - this name is fixed and will be used as a lookup during init
        num_classes = len(categories)

        # create a mapping from category to index
        category_to_index = {category: i for i, category in enumerate(categories)}

        dictionary = torch.zeros((num_classes, 2))
        for index, row in df.iterrows():
            key = row.iloc[0]
            value = [row.iloc[1], row.iloc[2]]
            if key in categories:
                (
                    dictionary[category_to_index[key], 0],
                    dictionary[category_to_index[key], 1],
                ) = np.radians(row.iloc[1]), np.radians(row.iloc[2])

        # Save the PyTorch tensor to a .pt file
        output_file = join(save_path, "index_to_gps_" + class_name + ".pt")
        torch.save(dictionary, output_file)

    train = pd.read_csv(join(data_path, "train.csv"), low_memory=False).fillna(
        "NaN"
    )

    u = train.groupby("unique_city").sample(n=1)

    country_df = (
        u.pivot(index="unique_city", columns="unique_country", values="unique_city")
        .notna()
        .astype(int)
        .fillna(0)
    )
    country_to_idx = {
        category: i for i, category in enumerate(list(country_df.columns))
    }
    city_country_matrix = torch.tensor(country_df.values) / 1.0

    region_df = (
        u.pivot(index="unique_city", columns="unique_region", values="unique_city")
        .notna()
        .astype(int)
        .fillna(0)
    )
    region_to_idx = {category: i for i, category in enumerate(list(region_df.columns))}
    city_region_matrix = torch.tensor(region_df.values) / 1.0

    country_df = (
        u.pivot(index="unique_city", columns="unique_country", values="unique_city")
        .notna()
        .astype(int)
        .fillna(0)
    )
    country_to_idx = {
        category: i for i, category in enumerate(list(country_df.columns))
    }
    city_country_matrix = torch.tensor(country_df.values) / 1.0

    output_file = join(save_path, "city_to_country.pt")
    torch.save(city_country_matrix, output_file)

    output_file = join(save_path, "country_to_idx.pt")
    torch.save(country_to_idx, output_file)

    region_df = (
        u.pivot(index="unique_city", columns="unique_region", values="unique_city")
        .notna()
        .astype(int)
        .fillna(0)
    )
    region_to_idx = {category: i for i, category in enumerate(list(region_df.columns))}
    city_region_matrix = torch.tensor(region_df.values) / 1.0

    output_file = join(save_path, "city_to_region.pt")
    torch.save(city_region_matrix, output_file)

    output_file = join(save_path, "region_to_idx.pt")
    torch.save(region_to_idx, output_file)

    area_df = (
        u.pivot(index="unique_city", columns="unique_sub-region", values="unique_city")
        .notna()
        .astype(int)
        .fillna(0)
    )
    area_to_idx = {category: i for i, category in enumerate(list(area_df.columns))}
    city_area_matrix = torch.tensor(area_df.values) / 1.0

    output_file = join(save_path, "city_to_area.pt")
    torch.save(city_area_matrix, output_file)

    output_file = join(save_path, "area_to_idx.pt")
    torch.save(area_to_idx, output_file)
    gt = torch.load(join(save_path, f"id_to_quad_{cfg.depth}_{cfg.do_split}.pt"))
    matrixes = []
    dicts = []
    for i in range(1, cfg.depth):
        # Step 2: Truncate strings to size cfg.depth - 1
        l = [s[: cfg.depth - i] if len(s) >= cfg.depth + 1 - i else s for s in gt]

        # Step 3: Get unique values in the modified list l
        h = list(set(l))

        # Step 4: Create a dictionary to map unique values to their index
        h_dict = {value: index for index, value in enumerate(h)}
        dicts.append(h_dict)

        # Step 5: Initialize a torch matrix with zeros
        matrix = torch.zeros((len(gt), len(h)))

        # Step 6: Fill in the matrix with 1s based on the mapping
        for h in range(len(gt)):
            j = h_dict[l[h]]
            matrix[h, j] = 1
        matrixes.append(matrix)

    output_file = join(save_path, "quadtree_matrixes.pt")
    torch.save(matrixes, output_file)

    output_file = join(save_path, "quadtree_dicts.pt")
    torch.save(dicts, output_file)


if __name__ == "__main__":
    main()
