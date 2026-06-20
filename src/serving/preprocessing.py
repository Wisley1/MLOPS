"""ClearML Serving preprocessing for DistilBERT arXiv classifier (Stage 4)."""

from __future__ import annotations

from typing import Any

import numpy as np
from transformers import AutoTokenizer, PreTrainedTokenizer, TensorType

ID2LABEL = {
    0: "art",
    1: "computer vision",
    2: "food",
    3: "games",
    4: "medicine",
    5: "microbiome",
    6: "physics",
    7: "transformers",
}

MAX_LENGTH = 256
TOKENIZER_NAME = "distilbert-base-uncased"


class Preprocess:
    """Pre/post-processing for ClearML Triton inference service."""

    def __init__(self) -> None:
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    def preprocess(
        self,
        body: dict,
        state: dict,
        collect_custom_statistics_fn=None,
    ) -> Any:
        text = body.get("text", "")
        tokens = self.tokenizer(
            text=text,
            max_length=MAX_LENGTH,
            truncation=True,
            padding="max_length",
            return_tensors=TensorType.NUMPY,
        )
        return [
            tokens["input_ids"].tolist(),
            tokens["attention_mask"].tolist(),
        ]

    def postprocess(
        self,
        data: Any,
        state: dict,
        collect_custom_statistics_fn=None,
    ) -> dict:
        logits = np.array(data)
        if logits.ndim == 2:
            pred_id = int(np.argmax(logits, axis=-1)[0])
        else:
            pred_id = int(np.argmax(logits))

        label = ID2LABEL.get(pred_id, str(pred_id))
        probabilities = _softmax(logits.reshape(-1)[: len(ID2LABEL)])

        return {
            "label": label,
            "class_id": pred_id,
            "probabilities": {
                ID2LABEL[i]: float(probabilities[i]) for i in range(len(ID2LABEL))
            },
        }


def _softmax(x: np.ndarray) -> np.ndarray:
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()
