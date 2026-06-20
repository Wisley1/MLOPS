"""Upload arXiv training CSV to ClearML Dataset (Stage 1)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from clearml import Dataset

from src.config import DATASET_NAME, DATASET_PROJECT, DATASET_VERSION


def build_text(row: pd.Series) -> str:
    title = str(row.get("title", "")).strip()
    abstract = str(row.get("abstract", "")).strip()
    if title and abstract:
        return f"{title}. {abstract}"
    return title or abstract


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload arXiv dataset to ClearML")
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=Path("arxiv_multiclass_classifier/arxiv_train_multiclass.csv"),
    )
    parser.add_argument(
        "--meta-path",
        type=Path,
        default=Path("arxiv_multiclass_classifier/arxiv_train_multiclass.meta.json"),
    )
    parser.add_argument("--max-rows", type=int, default=3000, help="Limit rows for upload")
    parser.add_argument("--dataset-version", type=str, default=DATASET_VERSION)
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip histogram/table logging (useful if ClearML API is unstable)",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path, nrows=args.max_rows)
    df["text"] = df.apply(build_text, axis=1)
    df = df[["text", "topic"]].dropna()

    output_dir = Path("data/prepared")
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared_csv = output_dir / "train.csv"
    df.to_csv(prepared_csv, index=False)

    meta = json.loads(args.meta_path.read_text(encoding="utf-8"))
    meta_path = output_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    dataset = Dataset.create(
        dataset_name=DATASET_NAME,
        dataset_project=DATASET_PROJECT,
        dataset_version=args.dataset_version,
        description="ArXiv multi-topic article classification dataset",
    )
    dataset.add_files(str(output_dir))
    dataset.add_tags(["arxiv", "text-classification", "multiclass"])

    if not args.skip_plots:
        try:
            topic_counts = df["topic"].value_counts()
            dataset.get_logger().report_histogram(
                title="Topic distribution",
                series="Topic distribution",
                values=topic_counts.tolist(),
                xlabels=topic_counts.index.tolist(),
                yaxis="Number of samples",
            )
            dataset.get_logger().report_table(
                "Dataset Preview",
                "Preview",
                table_plot=df.head(10),
            )
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: failed to log plots to ClearML ({exc}). Continuing upload.")

    dataset.upload()
    try:
        dataset.finalize()
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: finalize() failed ({exc}). Files may already be uploaded — check UI.")

    print(f"Dataset uploaded: id={dataset.id}, version={args.dataset_version}")
    print(f"Local path: {dataset.get_local_copy()}")


if __name__ == "__main__":
    main()
