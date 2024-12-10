import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import argparse
import os

from jean_zay.launch import JeanZayExperiment


def parse_mode():
    parser = argparse.ArgumentParser(
        description="Extract embeddings from a dataset using DINOv2"
    )
    parser.add_argument(
        "--launch",
        action="store_true",
        help="Launch the experiment",
    )
    parser.add_argument("--src_json", help="path to src json")
    parser.add_argument("--dest", help="path to dest")
    parser.add_argument(
        "--num_samples_per_tar",
        help="number of samples per tar",
        type=int,
        default=10000,
    )
    parser.add_argument("--number_of_jobs", help="number of jobs", type=int, default=10)
    args = parser.parse_args()

    return args


args = parse_mode()

cmd_modifiers = []
exps = []

exp_name = f"inaturalist_preprocessing"
job_name = f"inaturalist_preprocessing"
jz_exp = JeanZayExperiment(
    exp_name,
    job_name,
    slurm_array_nb_jobs=args.number_of_jobs,
    cmd_path="data/to_webdataset/inaturalist_to_wds.py",
    num_nodes=1,
    num_gpus_per_node=1,
    qos="t3",
    account="syq",
    gpu_type="v100",
    time="1:00:00",
)

exps.append(jz_exp)

trainer_modifiers = {}

exp_modifier = {
    "--src_json": args.src_json,
    "--dest": args.dest,
    "--num_samples_per_tar": args.num_samples_per_tar,
    "--number_of_jobs": args.number_of_jobs,
    "--job_offset": "${SLURM_ARRAY_TASK_ID}",
}

cmd_modifiers.append(dict(trainer_modifiers, **exp_modifier))


if __name__ == "__main__":
    for exp, cmd_modifier in zip(exps, cmd_modifiers):
        exp.build_cmd(cmd_modifier)
        if args.launch == True:
            exp.launch()
