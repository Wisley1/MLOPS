"""Publish the best trained model to ClearML Model Registry (Stage 3)."""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

from clearml import OutputModel, Task


def _read_metric(task, model, name: str) -> str | None:
    metadata = model.get_metadata(name)
    if metadata is not None:
        return str(metadata)

    scalars = task.get_last_scalar_metrics() or {}
    summary = scalars.get("Summary", {})
    value = summary.get(name)
    if isinstance(value, dict):
        return str(value.get("last", value.get("value")))
    if value is not None:
        return str(value)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish model to ClearML Registry")
    parser.add_argument("--task-id", type=str, required=True, help="ClearML training task ID")
    parser.add_argument("--model-name", type=str, default="arxiv-distilbert")
    parser.add_argument("--tag", action="append", default=[], help="Tags for the model")
    parser.add_argument(
        "--for-triton",
        action="store_true",
        help="Also publish an ONNX-bundle model (framework=ONNX) for Triton serving",
    )
    args = parser.parse_args()

    task = Task.get_task(task_id=args.task_id)
    model = task.models["output"][0]

    tags = args.tag or ["production", "distilbert", "arxiv"]
    model.tags = tags

    accuracy = _read_metric(task, model, "accuracy")
    f1 = _read_metric(task, model, "f1")
    if accuracy is not None:
        model.set_metadata("accuracy", accuracy, "metrics")
    if f1 is not None:
        model.set_metadata("f1", f1, "metrics")

    if model.name != args.model_name:
        model.name = args.model_name

    model.publish()

    print(f"Model published to Registry: {model.id}")
    print(f"Model name: {model.name}")
    if accuracy is not None:
        print(f"accuracy={accuracy}")
    if f1 is not None:
        print(f"f1={f1}")

    if args.for_triton:
        onnx_path = _find_onnx_artifact(task)
        if onnx_path is None:
            raise SystemExit(
                "ONNX weights not found on task. Expected output model .onnx file "
                "or artifact 'model_onnx'."
            )

        tmpdir = Path(tempfile.mkdtemp())
        onnx_dir = tmpdir / "onnx_bundle"
        onnx_dir.mkdir()
        shutil.copy2(onnx_path, onnx_dir / "model.onnx")

        triton_task = Task.init(
            project_name=task.project,
            task_name=f"{task.name}-triton-onnx",
            task_type=Task.TaskTypes.custom,
            output_uri=True,
        )
        triton_model = OutputModel(
            task=triton_task,
            framework="ONNX",
            name=f"{args.model_name}-onnx",
        )
        triton_model.update_weights(weights_filename=str(onnx_dir))
        triton_model.tags = tags + ["onnx"]
        if accuracy is not None:
            triton_model.set_metadata("accuracy", accuracy, "metrics")
        if f1 is not None:
            triton_model.set_metadata("f1", f1, "metrics")
        triton_task.flush(wait_for_uploads=True)
        triton_model.publish()
        triton_task.close()
        print(f"Triton ONNX model published: {triton_model.id}")
        print(f"Triton ONNX model url: {triton_model.url}")


def _find_onnx_artifact(task: Task) -> Path | None:
    output_model = task.models["output"][0]
    local = Path(output_model.get_local_copy())
    if local.is_file() and local.suffix == ".onnx" and local.stat().st_size > 1024:
        return local

    artifacts = task.artifacts or {}
    artifact = artifacts.get("model_onnx")
    if artifact is None:
        return None
    local = Path(artifact.get_local_copy())
    if local.exists() and local.stat().st_size > 1024:
        return local
    return None


if __name__ == "__main__":
    main()
