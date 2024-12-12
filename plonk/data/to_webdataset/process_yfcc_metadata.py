import dask
import dask.dataframe as dd
from dask.diagnostics import ProgressBar

with ProgressBar():
    ddf = dd.read_csv(
        "../datasets/YFCC100M/yfcc100m_dataset",
        names=[
            "photo_id",
            "user_nsid",
            "user_nickname",
            "date_taken",
            "date_uploaded",
            "capture_device",
            "title",
            "description",
            "user_tags",
            "machine_tags",
            "longitude",
            "latitude",
            "accuracy",
            "page_url",
            "download_url",
            "license_name",
            "license_url",
            "server_id",
            "farm_id",
            "secret",
            "secret_original",
            "extension",
            "media_type",
        ],
        dtype={
            "photo_id": str,
            "user_nsid": str,
            "user_nickname": str,
            "user_tags": str,
            "machine_tags": str,
            "longitude": float,
            "latitude": float,
            "accuracy": float,
            "server_id": str,
            "farm_id": str,
            "secret": str,
            "secret_original": str,
            "extension": str,
            "media_type": float,
        },
        sep="\t",
    )
    ddf = ddf[
        [
            "photo_id",
            "longitude",
            "latitude",
            "accuracy",
            "extension",
            "download_url",
            "media_type",
        ]
    ]
    filtered_ddf = ddf[
        ddf["longitude"].notnull()
        & ddf["latitude"].notnull()
        & (ddf["media_type"] == 0)
    ]
    del ddf["media_type"]
    hash_ddf = dd.read_csv(
        "../datasets/YFCC100M/yfcc100m_hash",
        names=["photo_id", "hash"],
        dtype={"photo_id": str, "hash": str},
        sep="\t",
    )
    filtered_ddf = filtered_ddf.merge(hash_ddf, on="photo_id", how="left")
    # Read the 4k photo IDs
    with open("../datasets/YFCC100M/yfcc_4k_ids.txt", "r") as f:
        test_photo_ids = set(f.read().splitlines())

    # Split the dataframe based on whether photo_id is in test set
    filter = filtered_ddf["photo_id"].isin(test_photo_ids)
    test_ddf = filtered_ddf[filter]
    train_ddf = filtered_ddf[~filter]

    train_ddf = train_ddf[train_ddf["accuracy"] >= 12]

    # Save the split dataframes
    test_ddf.to_csv(
        "../datasets/YFCC100M/yfcc_4k_dataset_with_gps.csv",
        sep="\t",
        index=False,
        single_file=True,
    )
    train_ddf = train_ddf.repartition(npartitions=len(train_ddf) // 100000 + 1)
    train_ddf.to_csv(
        "../datasets/YFCC100M/yfcc100m_dataset_with_gps_train/*.csv",
        sep="\t",
        index=False,
        single_file=False,
    )
