"""Streamlit UI for arXiv topic classification via ClearML Serving HTTP endpoint (Stage 5)."""

from __future__ import annotations

import os
import time

import requests
import streamlit as st

DEFAULT_SERVING_URL = os.getenv("CLEARML_SERVING_URL", "http://localhost:8088")
DEFAULT_ENDPOINT = os.getenv("CLEARML_SERVING_ENDPOINT", "arxiv_classify")
DEFAULT_VERSION = os.getenv("CLEARML_SERVING_VERSION", "1")

SAMPLE_TEXTS = {
    "art": (
        "This paper examines the evolution of impressionist painting techniques "
        "and their influence on contemporary abstract art in European galleries."
    ),
    "computer vision": (
        "We introduce a convolutional neural network for real-time object detection "
        "in autonomous driving scenarios with state-of-the-art mAP on COCO."
    ),
    "food": (
        "We study fermentation kinetics and flavor development in artisan sourdough "
        "bread baking using controlled temperature and hydration experiments."
    ),
    "games": (
        "We present a reinforcement learning agent for real-time strategy games "
        "with hierarchical planning and opponent modeling in multiplayer settings."
    ),
    "medicine": (
        "This clinical study evaluates the efficacy of a new immunotherapy treatment "
        "for patients with advanced melanoma using randomized controlled trials."
    ),
    "microbiome": (
        "We analyze gut microbiome composition and its association with inflammatory "
        "bowel disease using shotgun metagenomic sequencing of patient cohorts."
    ),
    "physics": (
        "High-beta optics calculus at IP2 for forward physics in LHC Run 3. "
        "We present beam optics measurements and luminosity optimization "
        "for proton-proton collisions at the LHC interaction point."
    ),
    "transformers": (
        "We propose a novel transformer architecture for long-range sequence modeling "
        "with linear attention complexity and improved training stability."
    ),
}


def predict(text: str, serving_url: str, endpoint: str, version: str) -> tuple[dict, float]:
    url = f"{serving_url.rstrip('/')}/serve/{endpoint}/{version}"
    started = time.perf_counter()
    response = requests.post(
        url,
        json={"text": text},
        headers={"accept": "application/json", "Content-Type": "application/json"},
        timeout=60,
    )
    latency_ms = (time.perf_counter() - started) * 1000
    response.raise_for_status()
    return response.json(), latency_ms


def main() -> None:
    st.set_page_config(page_title="ArXiv Topic Classifier", page_icon="📄", layout="centered")
    st.title("ArXiv Topic Classifier")
    st.caption("UI calls ClearML Serving over HTTP — model is not loaded locally.")

    with st.sidebar:
        st.header("Endpoint settings")
        serving_url = st.text_input("Serving URL", value=DEFAULT_SERVING_URL)
        endpoint = st.text_input("Endpoint name", value=DEFAULT_ENDPOINT)
        version = st.text_input("Model version", value=DEFAULT_VERSION)

    st.subheader("Input")
    sample_key = st.selectbox("Load sample text", ["—"] + list(SAMPLE_TEXTS.keys()))
    default_text = SAMPLE_TEXTS.get(sample_key, "") if sample_key != "—" else ""

    text = st.text_area(
        "Article text (title + abstract)",
        value=default_text,
        height=200,
        placeholder="Paste article title and abstract here...",
    )

    if st.button("Predict", type="primary", disabled=not text.strip()):
        try:
            with st.spinner("Calling inference endpoint..."):
                result, latency_ms = predict(text, serving_url, endpoint, version)

            label = result.get("label") or result.get("y") or result.get("data")
            st.success(f"**Predicted topic:** {label}")
            st.metric("Latency", f"{latency_ms:.1f} ms")

            if "probabilities" in result:
                st.subheader("Class probabilities")
                for topic, prob in sorted(
                    result["probabilities"].items(),
                    key=lambda item: item[1],
                    reverse=True,
                ):
                    st.progress(min(max(prob, 0.0), 1.0), text=f"{topic}: {prob:.2%}")

            with st.expander("Raw response"):
                st.json(result)

        except requests.ConnectionError:
            st.error(
                f"Cannot connect to serving endpoint at `{serving_url}`. "
                "Make sure ClearML Serving is running and the URL is correct."
            )
        except requests.HTTPError as exc:
            st.error(f"HTTP error {exc.response.status_code}: {exc.response.text}")
        except requests.Timeout:
            st.error("Request timed out. The serving service may be overloaded or unavailable.")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Unexpected error: {exc}")


if __name__ == "__main__":
    main()
