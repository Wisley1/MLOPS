"""Fine-tune DistilBERT for arXiv topic classification via ClearML Agent (Stage 2)."""

from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from clearml import Dataset, Logger, OutputModel, Task
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset as TorchDataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from src.config import (
    BASE_MODEL,
    DATASET_NAME,
    DATASET_PROJECT,
    MAX_LENGTH,
    QUEUE_NAME,
    TOPICS,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _git_branch(project_root: Path) -> str:
    import subprocess

    return subprocess.check_output(
        ["git", "-C", str(project_root), "rev-parse", "--abbrev-ref", "HEAD"],
        text=True,
    ).strip()


def _configure_remote_execution(task: Task, queue: str) -> None:
    """Point ClearML Agent at the MLOPS folder, not the parent ~/VSCode git repo."""
    if not (PROJECT_ROOT / ".git").exists():
        raise RuntimeError(
            "MLOPS needs its own git repo for ClearML Agent. Run: ./scripts/init_git.sh"
        )

    if platform.system() == "Darwin":
        task.set_base_docker(docker_image="", docker_arguments="")

    task.set_script(
        repository=PROJECT_ROOT.resolve().as_uri(),
        branch=_git_branch(PROJECT_ROOT),
        commit="",
        diff="",
        working_dir=".",
        entry_point="src/train.py",
    )
    task.execute_remotely(queue_name=queue)


class TextClassificationDataset(TorchDataset):
    def __init__(self, texts, labels, tokenizer, max_length: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    return accuracy, f1, np.array(all_labels), np.array(all_preds)


def plot_confusion_matrix(y_true, y_pred, labels, output_path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def export_onnx(model, tokenizer, output_path: Path, max_length: int) -> None:
    class OnnxWrapper(torch.nn.Module):
        def __init__(self, inner_model):
            super().__init__()
            self.inner_model = inner_model

        def forward(self, input_ids, attention_mask):
            return self.inner_model(input_ids=input_ids, attention_mask=attention_mask).logits

    wrapper = OnnxWrapper(model)
    wrapper.eval()
    dummy = tokenizer(
        "sample text for onnx export",
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )
    input_ids = dummy["input_ids"]
    attention_mask = dummy["attention_mask"]

    torch.onnx.export(
        wrapper,
        (input_ids, attention_mask),
        str(output_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "logits": {0: "batch"},
        },
        opset_version=14,
    )


def load_data_from_clearml(dataset_id: str | None, dataset_version: str) -> tuple[pd.DataFrame, dict]:
    if dataset_id:
        dataset = Dataset.get(dataset_id=dataset_id)
    else:
        dataset = Dataset.get(
            dataset_name=DATASET_NAME,
            dataset_project=DATASET_PROJECT,
            dataset_version=dataset_version,
        )

    local_path = Path(dataset.get_local_copy())
    train_csv = local_path / "train.csv"
    meta_path = local_path / "meta.json"

    df = pd.read_csv(train_csv)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return df, meta, dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Train arXiv topic classifier")
    parser.add_argument("--dataset-id", type=str, default=None)
    parser.add_argument("--dataset-version", type=str, default="1.0")
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=2000)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--experiment-name", type=str, default="distilbert-finetune")
    parser.add_argument(
        "--remote",
        action="store_true",
        help="Enqueue task to ClearML Agent queue instead of running locally",
    )
    parser.add_argument("--queue", type=str, default=QUEUE_NAME)
    args = parser.parse_args()

    Task.add_requirements("requirements.txt")
    task = Task.init(
        project_name="Arxiv Classification",
        task_name=args.experiment_name,
        task_type=Task.TaskTypes.training,
        output_uri=True,
        auto_connect_frameworks={"detect_repository": False},
    )

    hyperparams = {
        "dataset_id": args.dataset_id,
        "dataset_version": args.dataset_version,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "max_samples": args.max_samples,
        "test_size": args.test_size,
        "seed": args.seed,
        "base_model": BASE_MODEL,
        "max_length": MAX_LENGTH,
    }
    task.connect(hyperparams)

    if args.remote:
        _configure_remote_execution(task, args.queue)

    df, meta, dataset = load_data_from_clearml(args.dataset_id, args.dataset_version)
    task.set_parameter("clearml_dataset_id", dataset.id)

    topic_to_id = meta["topic_to_id"]
    id_to_topic = {v: k for k, v in topic_to_id.items()}

    if args.max_samples and len(df) > args.max_samples:
        df = df.sample(n=args.max_samples, random_state=args.seed)

    train_df, test_df = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=df["topic"],
    )

    train_texts = train_df["text"].tolist()
    train_labels = [topic_to_id[t] for t in train_df["topic"]]
    test_texts = test_df["text"].tolist()
    test_labels = [topic_to_id[t] for t in test_df["topic"]]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(TOPICS),
        id2label=id_to_topic,
        label2id=topic_to_id,
    )
    model.to(device)

    train_loader = DataLoader(
        TextClassificationDataset(train_texts, train_labels, tokenizer, MAX_LENGTH),
        batch_size=args.batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        TextClassificationDataset(test_texts, test_labels, tokenizer, MAX_LENGTH),
        batch_size=args.batch_size,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    logger: Logger = task.get_logger()
    output_model = OutputModel(task=task, framework="ONNX", name="arxiv-distilbert")

    model.train()
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / max(len(train_loader), 1)
        logger.report_scalar("train", "loss", avg_loss, iteration=epoch)
        print(f"Epoch {epoch + 1}/{args.epochs}, loss={avg_loss:.4f}")

    accuracy, f1, y_true, y_pred = evaluate(model, test_loader, device)
    logger.report_single_value("accuracy", accuracy)
    logger.report_single_value("f1", f1)
    task.get_logger().report_text(f"accuracy={accuracy:.4f}, f1={f1:.4f}")

    cm_path = Path("outputs/confusion_matrix.png")
    cm_path.parent.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(y_true, y_pred, TOPICS, cm_path)
    logger.report_image("evaluation", "confusion_matrix", iteration=0, local_path=str(cm_path))

    model_dir = Path("outputs/model")
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    onnx_path = Path("outputs/model.onnx")
    export_onnx(model, tokenizer, onnx_path, MAX_LENGTH)

    output_model.update_weights(weights_filename=str(onnx_path))
    output_model.update_labels(labels=topic_to_id)
    output_model.report_single_value("accuracy", accuracy)
    output_model.report_single_value("f1", f1)
    output_model.set_metadata("dataset_id", dataset.id, "dataset")
    output_model.set_metadata("base_model", BASE_MODEL, "training")
    output_model.set_metadata("accuracy", str(round(accuracy, 4)), "metrics")
    output_model.set_metadata("f1", str(round(f1, 4)), "metrics")

    task.upload_artifact("model_dir", artifact_object=str(model_dir))
    task.upload_artifact("model_onnx", artifact_object=str(onnx_path))

    print(f"Training complete: accuracy={accuracy:.4f}, f1={f1:.4f}")
    print(f"Task ID: {task.id}")


if __name__ == "__main__":
    main()
