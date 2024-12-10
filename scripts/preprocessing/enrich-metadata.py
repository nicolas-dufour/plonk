import os
import json
import joblib
import pandas as pd
import numpy as np
import reverse_geocoder
from os.path import join, dirname


class QuadTree(object):
    def __init__(
        self, data, mins=None, maxs=None, id="", depth=3, min_split=0, do_split=1000
    ):
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

        if depth > 0 and len(self.data) >= do_split:
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
            if data_q1.shape[0] > min_split:
                self.children.append(
                    QuadTree(data_q1, [xmin, ymin], [xmid, ymid], id + "0", depth - 1)
                )
            if data_q2.shape[0] > min_split:
                self.children.append(
                    QuadTree(data_q2, [xmin, ymid], [xmid, ymax], id + "1", depth - 1)
                )
            if data_q3.shape[0] > min_split:
                self.children.append(
                    QuadTree(data_q3, [xmid, ymin], [xmax, ymid], id + "2", depth - 1)
                )
            if data_q4.shape[0] > min_split:
                self.children.append(
                    QuadTree(data_q4, [xmid, ymid], [xmax, ymax], id + "3", depth - 1)
                )

    def unwrap(self):
        if len(self.children) == 0:
            return {self.id: [self.mins, self.maxs, self.data.copy()]}
        else:
            d = dict()
            for child in self.children:
                d.update(child.unwrap())
            return d


def extract(qt):
    cluster = qt.unwrap()
    boundaries, data = {}, []
    for id, vs in cluster.items():
        (min_lat, min_lon), (max_lat, max_lon), points = vs
        points["category"] = id
        data.append(points)
        boundaries[id] = (
            float(min_lat),
            float(min_lon),
            float(max_lat),
            float(max_lon),
        )

    data = pd.concat(data)
    return boundaries, data


if __name__ == "__main__":
    # merge into one DataFrame
    data_path = join(dirname(dirname(__file__)), "datasets", "osv5m")
    train_fp = join(data_path, f"train.csv")
    test_fp = join(data_path, f"test.csv")

    df_train = pd.read_csv(train_fp)
    df_train["split"] = "train"

    df_test = pd.read_csv(test_fp)
    df_test["split"] = "test"

    df = pd.concat([df_train, df_test])
    size_before = df.shape[0]
    qt = QuadTree(df, depth=15)
    boundaries, df = extract(qt)
    assert df.shape[0] == size_before

    location = reverse_geocoder.search(
        [(lat, lon) for lat, lon in zip(df["latitude"], df["longitude"])]
    )
    df["city"] = [l.get("name", "") for l in location]
    df["country"] = [l.get("cc", "") for l in location]
    del location

    df_train = df[df["split"] == "train"].drop(["split"], axis=1)
    df_test = df[df["split"] == "test"].drop(["split"], axis=1)
    assert (df_train.shape[0] + df_test.shape[0]) == size_before

    json.dump(boundaries, open(join(data_path, "borders.json"), "w"))
    df_train.to_csv(train_fp, index=False)
    df_test.to_csv(test_fp, index=False)
