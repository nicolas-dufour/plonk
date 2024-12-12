import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import argparse
import os

from jean_zay.launch import JeanZayExperiment


def parse_mode():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--src", help="path to source files")
    parser.add_argument("--dest", help="path to destination files")
    args = parser.parse_args()

    return args


args = parse_mode()

dataset_path = Path(args.src)

list_of_shards = list(dataset_path.glob("*.tar"))
list_of_shards.sort()


cmd_modifiers = []
exps = []

exp_name = f"preprocess_data"
job_name = f"preprocess_data"
jz_exp = JeanZayExperiment(
    exp_name,
    job_name,
    slurm_array_nb_jobs=len(list_of_shards),
    cmd_path="data/to_webdataset/osv_to_wds.py",
    num_nodes=1,
    qos="t3",
    account="syq",
    gpu_type="a100",
    time="01:00:00",
)

exps.append(jz_exp)

trainer_modifiers = {}

exp_modifier = {
    "--src": dataset_path,
    "--dest": Path(args.dest),
    "--shard_id": "${SLURM_ARRAY_TASK_ID}",
}

cmd_modifiers.append(dict(trainer_modifiers, **exp_modifier))


if __name__ == "__main__":
    for exp, cmd_modifier in zip(exps, cmd_modifiers):
        exp.build_cmd(cmd_modifier)
        if args.launch == True:
            exp.launch()
