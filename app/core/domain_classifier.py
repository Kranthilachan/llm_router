"""
Semantic-embedding domain classifier.

Loads a pre-trained ONNX embedding model at import time (via `fastembed`,
which uses onnxruntime — no torch dependency) and encodes every prompt in
data/domain_examples.json into a 384-dim vector. Per query, we encode the
incoming prompt and return the domain whose nearest-neighbor cosine
similarity is highest, along with that similarity as the confidence.

No model training happens here — the pre-trained model is used only for
inference. The stored embeddings are just a labeled lookup table.

Public interface (kept identical to the previous TF-IDF version so that
callers in app/core/semantic_parser.py don't need to change):

    classify_domain(prompt) -> (label, similarity)
    LEARNED_DOMAIN_CONFIDENCE_THRESHOLD  — float, minimum similarity to
                                           override keyword heuristics
    RISK_METADATA                         — per-domain risk_tier/type/doc_type

Graceful degradation: if fastembed is unavailable or the dataset file is
missing, classify_domain() returns (None, 0.0) and the parser falls back
to keyword heuristics.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

DATASET_PATH = Path(__file__).resolve().parents[2] / "data" / "domain_examples.json"

# Minimum cosine similarity required for the embedding classifier to win.
# Below this the parser either falls through to keyword heuristics OR
# escalates to the LLM tiebreaker (when llm_parser is wired in route()).
LEARNED_DOMAIN_CONFIDENCE_THRESHOLD = 0.55

RISK_METADATA: Dict[str, Tuple[str, str, str]] = {
    "crm": ("low", "standard", "crm_record"),
    "hrm": ("medium", "pii_sensitive", "hr_record"),
    "project": ("low", "standard", "project_document"),
    "accounts": ("medium", "operational", "accounting_record"),
}


_model = None
_embeddings_by_domain: Dict[str, np.ndarray] = {}
_labels: List[str] = []


def _load_dataset(path: Path) -> Optional[Dict[str, List[str]]]:
    if not path.exists():
        logger.warning("Domain dataset not found at %s; classifier disabled.", path)
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to load domain dataset at %s: %s", path, exc)
        return None
    if not isinstance(data, dict) or not data:
        logger.warning("Domain dataset at %s is empty or malformed.", path)
        return None
    return {label: prompts for label, prompts in data.items() if prompts}


def _load_model():
    try:
        # Lazy import so the module still loads when fastembed isn't
        # installed — classify_domain then returns (None, 0.0).
        from fastembed import TextEmbedding
    except ImportError:
        logger.warning("fastembed unavailable; classifier disabled.")
        return None

    from app.config import ATLAS_EMBEDDING_MODEL
    try:
        return TextEmbedding(model_name=ATLAS_EMBEDDING_MODEL)
    except Exception as exc:  # network / download / model load errors
        logger.warning("Failed to load embedding model %s: %s", model_name, exc)
        return None


def _encode(model, prompts: List[str]) -> np.ndarray:
    """Return an (N, dim) L2-normalized matrix for a list of prompts."""
    vecs = np.asarray(list(model.embed(prompts)), dtype=np.float32)
    # fastembed's BGE / MiniLM ONNX models return already-normalized vectors;
    # renormalize defensively so cosine == dot product regardless of model.
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


def _bootstrap() -> None:
    global _model, _embeddings_by_domain, _labels

    dataset = _load_dataset(DATASET_PATH)
    if dataset is None:
        return

    model = _load_model()
    if model is None:
        return

    embeddings_by_domain: Dict[str, np.ndarray] = {}
    for domain, prompts in dataset.items():
        embeddings_by_domain[domain] = _encode(model, prompts)

    _model = model
    _embeddings_by_domain = embeddings_by_domain
    _labels = sorted(embeddings_by_domain.keys())

    # Warm up the encoder so the first live request doesn't pay init cost.
    try:
        _encode(_model, ["warmup"])
    except Exception:
        pass


_bootstrap()


def classify_domain(prompt: str) -> Tuple[Optional[str], float]:
    """Return (domain_label, cosine_similarity) or (None, 0.0) if disabled."""
    if _model is None or not _embeddings_by_domain or not prompt.strip():
        return None, 0.0

    query_vec = _encode(_model, [prompt])[0]

    best_domain: Optional[str] = None
    best_score = -1.0
    for domain, matrix in _embeddings_by_domain.items():
        # matrix @ query_vec  →  cosine similarity per training example
        # (both are L2-normalized already)
        sims = matrix @ query_vec
        top = float(sims.max())
        if top > best_score:
            best_score = top
            best_domain = domain

    return best_domain, max(best_score, 0.0)


def available_domains() -> List[str]:
    return list(_labels)
