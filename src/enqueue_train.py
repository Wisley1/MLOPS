"""Enqueue training experiments to ClearML Agent queue."""

from __future__ import annotations

import argparse
import subprocess
import sys

from src.config import PROJECT_NAME, QUEUE_NAME


EXPERIMENTS = [
    {
        "task_name": "exp1-lr2e5-epoch1",
        "learning_rate": 2e-5,
        "epochs": 3,
        "batch_size": 16,
    },
    {
        "task_name": "exp2-lr5e5-epoch2",
        "learning_rate": 5e-5,
        "epochs": 5,
        "batch_size": 8,
    },
]


def enqueue_experiment(
    experiment: dict,
    dataset_id: str,
    dataset_version: str,
    queue: str,
) -> None:
    cmd = [
        sys.executable,
        "-m",
        "src.train",
        "--remote",
        "--queue",
        queue,
        "--dataset-id",
        dataset_id,
        "--dataset-version",
        dataset_version,
        "--experiment-name",
        experiment["task_name"],
        "--learning-rate",
        str(experiment["learning_rate"]),
        "--epochs",
        str(experiment["epochs"]),
        "--batch-size",
        str(experiment["batch_size"]),
    ]
    print(f"Enqueuing {experiment['task_name']} -> queue={queue}")
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Enqueue training experiments")
    parser.add_argument("--dataset-id", type=str, required=True)
    parser.add_argument("--dataset-version", type=str, default="1.0")
    parser.add_argument("--queue", type=str, default=QUEUE_NAME)
    parser.add_argument("--experiment", type=int, choices=[1, 2], default=None)
    args = parser.parse_args()

    experiments = EXPERIMENTS
    if args.experiment is not None:
        experiments = [EXPERIMENTS[args.experiment - 1]]

    for exp in experiments:
        enqueue_experiment(
            experiment=exp,
            dataset_id=args.dataset_id,
            dataset_version=args.dataset_version,
            queue=args.queue,
        )

    print(f"Done. Check UI -> Projects -> {PROJECT_NAME}")


if __name__ == "__main__":
    main()
