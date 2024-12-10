import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import argparse
import os

from jean_zay.launch import JeanZayExperiment


def parse_mode():
    parser = argparse.ArgumentParser(
        description="Extract embeddings from YFCC dataset using DINOv2"
    )
    parser.add_argument(
        "--launch",
        action="store_true",
        help="Launch the experiment",
    )
    parser.add_argument("--src_csv_dir", help="path to source csv directory")
    parser.add_argument("--src_images_dir", help="path to source images directory")
    parser.add_argument("--dest", help="path to destination")
    parser.add_argument(
        "--num_samples_per_tar",
        help="number of samples per tar",
        type=int,
        default=10000,
    )
    parser.add_argument("--batch_size", help="batch size", type=int, default=256)
    args = parser.parse_args()

    return args


args = parse_mode()

number_of_jobs = len(list(Path(args.src_csv_dir).glob("*.csv")))
cmd_modifiers = []
exps = []

exp_name = f"yfcc_preprocessing"
job_name = f"yfcc_preprocessing"
jz_exp = JeanZayExperiment(
    exp_name,
    job_name,
    slurm_array_nb_jobs=number_of_jobs,
    cmd_path="data/to_webdataset/yfcc_to_wds.py",
    num_nodes=1,
    num_gpus_per_node=1,
    qos="t3",
    account="syq",
    gpu_type="a100",
    time="1:30:00",
)

exps.append(jz_exp)

trainer_modifiers = {}

exp_modifier = {
    "--src_csv_dir": args.src_csv_dir,
    "--src_images_dir": args.src_images_dir,
    "--dest": args.dest,
    "--num_samples_per_tar": args.num_samples_per_tar,
    "--job_offset": "${SLURM_ARRAY_TASK_ID}",
    "--batch_size": args.batch_size,
}

cmd_modifiers.append(dict(trainer_modifiers, **exp_modifier))


if __name__ == "__main__":
    for exp, cmd_modifier in zip(exps, cmd_modifiers):
        exp.build_cmd(cmd_modifier)
        if args.launch == True:
            exp.launch()
