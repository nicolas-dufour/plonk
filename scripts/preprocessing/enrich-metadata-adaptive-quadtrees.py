import hydra
import torch
import numpy as np
import pandas as pd
import statistics
from os.path import join, dirname
import matplotlib.pyplot as plt


class QuadTree(object):
    def __init__(self, data, id="", depth=3, do_split=5000):
        self.id = id
        self.data = data

        coord = data[["latitude", "longitude"]].to_numpy()

        # if mins is None:
        mins = coord.min(0)
        # if maxs is None:
        maxs = coord.max(0)

        self.mins = np.asarray(mins)
        self.maxs = np.asarray(maxs)
        self.sizes = self.maxs - self.mins

        self.children = []

        # sort by latitude
        sorted_data_lat = sorted(coord, key=lambda point: point[0])

        # get the median lat
        median_lat = statistics.median(point[0] for point in sorted_data_lat)

        # Divide the cell into two half-cells based on the median lat
        data_left = [point for point in sorted_data_lat if point[0] <= median_lat]
        data_right = [point for point in sorted_data_lat if point[0] > median_lat]

        # Sort the data points by long in each half-cell
        sorted_data_left_lon = sorted(data_left, key=lambda point: point[1])
        sorted_data_right_lon = sorted(data_right, key=lambda point: point[1])

        # Calculate the median ylong coordinate in each half-cell
        median_lon_left = statistics.median(point[1] for point in sorted_data_left_lon)
        median_lon_right = statistics.median(
            point[1] for point in sorted_data_right_lon
        )

        if (depth > 0) and (len(self.data) >= do_split):
            # split the data into four quadrants
            data_q1 = data[
                (data["latitude"] < median_lat) & (data["longitude"] < median_lon_left)
            ]
            data_q2 = data[
                (data["latitude"] < median_lat) & (data["longitude"] >= median_lon_left)
            ]
            data_q3 = data[
                (data["latitude"] >= median_lat)
                & (data["longitude"] < median_lon_right)
            ]
            data_q4 = data[
                (data["latitude"] >= median_lat)
                & (data["longitude"] >= median_lon_right)
            ]

            # recursively build a quad tree on each quadrant which has data
            if data_q1.shape[0] > 0:
                self.children.append(
                    QuadTree(
                        data_q1,
                        id + "0",
                        depth - 1,
                        do_split=do_split,
                    )
                )
            if data_q2.shape[0] > 0:
                self.children.append(
                    QuadTree(
                        data_q2,
                        id + "1",
                        depth - 1,
                        do_split=do_split,
                    )
                )
            if data_q3.shape[0] > 0:
                self.children.append(
                    QuadTree(
                        data_q3,
                        id + "2",
                        depth - 1,
                        do_split=do_split,
                    )
                )
            if data_q4.shape[0] > 0:
                self.children.append(
                    QuadTree(
                        data_q4,
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
    return boundaries, data


def vizu(name_new_column, df_train, boundaries, do_split):
    plt.hist(df_train[name_new_column], bins=len(boundaries))
    plt.xlabel("Cluster ID")
    plt.ylabel("Number of images")
    plt.title("Cluster distribution")
    plt.yscale("log")
    plt.ylim(10, do_split)
    plt.savefig(f"{name_new_column}_distrib.png")
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
    plt.savefig(f"{name_new_column}_map.png")


@hydra.main(
    config_path="../configs/scripts",
    config_name="enrich-metadata-quadtree",
    version_base=None,
)
def main(cfg):

    data_path = join(cfg.data_dir, "osv5m")
    name_new_column = f"adaptive_quadtree_{cfg.depth}_{cfg.do_split}"

    # Create clusters from train images
    train_fp = join(data_path, f"train.csv")
    df_train = pd.read_csv(train_fp)

    qt = QuadTree(df_train, depth=cfg.depth, do_split=cfg.do_split)
    boundaries, df_train = extract(qt, name_new_column)

    vizu(name_new_column, df_train, boundaries, cfg.do_split)

    # Save clusters
    boundaries = pd.DataFrame.from_dict(
        boundaries,
        orient="index",
        columns=["min_lat", "min_lon", "max_lat", "max_lon", "mean_lat", "mean_lon"],
    )
    boundaries.to_csv(f"{name_new_column}.csv", index_label="cluster_id")

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
    coord = torch.stack([lat / 90, lon / 180], dim=-1)
    torch.save(
        coord,
        join(
            data_path, f"index_to_gps_adaptive_quadtree_{cfg.depth}_{cfg.do_split}.pt"
        ),
    )

    # Overwrite test.csv and train.csv
    if cfg.overwrite_csv:
        df_train.to_csv(train_fp, index=False)
        df_test.to_csv(test_fp, index=False)


if __name__ == "__main__":
    main()
