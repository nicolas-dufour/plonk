import webdataset as wds
from pathlib import Path
import json
import numpy as np
from PIL import Image


def main(
    src_json,
    dest_folder,
    num_samples_per_tar=10000,
    number_of_jobs=10,
    job_offset=0,
):
    with open(src_json, "r") as f:
        data = json.load(f)
    import pandas as pd

    root_path = Path(src_json).parent

    # Convert images list to pandas dataframe
    data_df = pd.DataFrame(data["images"])
    if "annotations" in data:
        has_annotations = True
        annotations_df = pd.DataFrame(data["annotations"])
        # Join the dataframes on id to get category_id from annotations
        data_df = data_df.merge(
            annotations_df[["id", "category_id"]],
            left_on="id",
            right_on="id",
            how="left",
        )
        categories_df = pd.DataFrame(data["categories"])
        data_df = data_df.merge(
            categories_df[
                [
                    "id",
                    "name",
                    "common_name",
                    "supercategory",
                    "kingdom",
                    "phylum",
                    "class",
                    "order",
                    "family",
                    "genus",
                    "specific_epithet",
                ]
            ],
            left_on="category_id",
            right_on="id",
            how="left",
        )
        data_df.rename(
            columns={
                "id_x": "id",
            },
            inplace=True,
        )
        del data_df["id_y"]
    else:
        has_annotations = False
    data_df = data_df[data_df["latitude"].notna() & data_df["longitude"].notna()]
    num_samples = len(data_df)
    num_total_tar = num_samples // num_samples_per_tar + (
        1 if num_samples % num_samples_per_tar > 0 else 0
    )
    number_of_tar_per_job = num_total_tar // number_of_jobs
    if job_offset == number_of_jobs - 1:
        data_df = data_df.iloc[
            number_of_tar_per_job * job_offset * num_samples_per_tar :
        ]
    else:
        data_df = data_df.iloc[
            number_of_tar_per_job
            * job_offset
            * num_samples_per_tar : number_of_tar_per_job
            * (job_offset + 1)
            * num_samples_per_tar
        ]
    print(f"Processing job {job_offset} with {len(data_df)} / {num_samples} samples")
    print(f"Number of tar: {number_of_tar_per_job} / {num_total_tar}")
    print(f"Start shard: {number_of_tar_per_job * job_offset}")
    with wds.ShardWriter(
        str(Path(dest_folder) / "%04d.tar"),
        maxcount=num_samples_per_tar,
        start_shard=number_of_tar_per_job * job_offset,
    ) as sink:
        for i in range(len(data_df)):
            row = data_df.iloc[i]
            image_path = Path(root_path) / Path("images") / row["file_name"]
            dinov2_embedding_path = (
                Path(root_path)
                / Path("embeddings")
                / Path("dinov2")
                / f"{row['file_name'].replace('.jpg', '.npy')}"
            )
            sample = {
                "__key__": str(row["id"]),
                "jpg": Image.open(image_path).convert("RGB"),
                "dinov2_vitl14_registers.npy": np.load(dinov2_embedding_path),
                "json": row.to_dict(),
            }
            sink.write(sample)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--src_json", help="pixel_input_folder")
    parser.add_argument("--dest", help="path to destination web")
    parser.add_argument(
        "--num_samples_per_tar",
        help="number of samples per tar",
        type=int,
        default=10000,
    )
    parser.add_argument("--number_of_jobs", help="number of jobs", type=int, default=10)
    parser.add_argument("--job_offset", help="job offset", type=int, default=0)
    args = parser.parse_args()

    dest = Path(args.dest)
    dest.mkdir(exist_ok=True, parents=True)

    main(
        args.src_json,
        args.dest,
        args.num_samples_per_tar,
        args.number_of_jobs,
        args.job_offset,
    )
