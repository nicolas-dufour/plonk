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
    parser.add_argument(
        "--number_of_splits",
        type=int,
        help="Number of splits to process",
        default=1,
    )
    parser.add_argument(
        "--input_path",
        type=str,
        help="Path to the input dataset",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Path to the output dataset",
    )
    args = parser.parse_args()

    return args


args = parse_mode()

cmd_modifiers = []
exps = []

exp_name = f"preprocess_data"
job_name = f"preprocess_data"
jz_exp = JeanZayExperiment(
    exp_name,
    job_name,
    slurm_array_nb_jobs=args.number_of_splits,
    cmd_path="data/extract_embeddings/dino_v2.py",
    num_nodes=1,
    num_gpus_per_node=1,
    qos="t3",
    account="mya",
    gpu_type="h100",
    time="02:00:00",
)

exps.append(jz_exp)

trainer_modifiers = {}

exp_modifier = {
    "--input_path": args.input_path,
    "--output_path": args.output_path,
    "--number_of_splits": args.number_of_splits,
    "--split_index": "${SLURM_ARRAY_TASK_ID}",
}

cmd_modifiers.append(dict(trainer_modifiers, **exp_modifier))


if __name__ == "__main__":
    for exp, cmd_modifier in zip(exps, cmd_modifiers):
        exp.build_cmd(cmd_modifier)
        if args.launch == True:
            exp.launch()
