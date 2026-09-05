import os
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np


def _enabled() -> bool:
    value = os.getenv("KGQA_LOCAL_EMBEDDINGS", "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _model_path() -> str:
    return (
        os.getenv("KGQA_LOCAL_EMBEDDING_MODEL", "")
        or os.getenv("LOCAL_EMBEDDING_MODEL", "")
        or os.getenv("SENTENCE_TRANSFORMERS_MODEL", "")
    ).strip()


@lru_cache(maxsize=1)
def _load_model():
    if not _enabled():
        return None

    model_path = _model_path()
    if not model_path:
        return None

    path = Path(model_path).expanduser()
    if not path.exists():
        # Keep the pipeline deterministic/offline: do not let sentence-transformers
        # download a model implicitly from a remote registry.
        return None

    try:
        from sentence_transformers import SentenceTransformer
    except Exception:
        return None

    try:
        return SentenceTransformer(str(path), local_files_only=True)
    except TypeError:
        return SentenceTransformer(str(path))
    except Exception:
        return None


@lru_cache(maxsize=32768)
def _embed_text(text: str) -> tuple[float, ...]:
    model = _load_model()
    if model is None:
        return ()

    clean_text = " ".join(str(text or "").split())
    if not clean_text:
        return ()

    try:
        vector = model.encode(
            clean_text,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
    except TypeError:
        vector = model.encode(clean_text, convert_to_numpy=True, show_progress_bar=False)
        norm = float(np.linalg.norm(vector))
        if norm > 0.0:
            vector = vector / norm
    except Exception:
        return ()

    arr = np.asarray(vector, dtype=float).reshape(-1)
    if arr.size == 0:
        return ()
    return tuple(float(v) for v in arr)


def local_embedding_cosine(text_a: str, text_b: str) -> float:
    """Return local sentence-transformer cosine similarity, or 0 when disabled.

    This intentionally never calls a remote embedding endpoint. The model must
    already exist on disk and be enabled through environment variables.
    """

    vec_a = _embed_text(text_a)
    vec_b = _embed_text(text_b)
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    return float(np.dot(np.asarray(vec_a, dtype=float), np.asarray(vec_b, dtype=float)))


def local_embedding_cosines(text_a: str, texts_b: Sequence[str]) -> list[float]:
    """Return cosine similarities from one text to many texts using batch encode."""

    model = _load_model()
    if model is None:
        return [0.0 for _ in texts_b]

    vec_a = _embed_text(text_a)
    if not vec_a:
        return [0.0 for _ in texts_b]

    clean_texts = [" ".join(str(text or "").split()) for text in texts_b]
    if not clean_texts:
        return []

    try:
        matrix = model.encode(
            clean_texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
    except TypeError:
        matrix = model.encode(clean_texts, convert_to_numpy=True, show_progress_bar=False)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        matrix = matrix / norms
    except Exception:
        return [0.0 for _ in texts_b]

    arr_a = np.asarray(vec_a, dtype=float).reshape(-1)
    arr_b = np.asarray(matrix, dtype=float)
    if arr_b.ndim == 1:
        arr_b = arr_b.reshape(1, -1)
    if arr_b.shape[1] != arr_a.shape[0]:
        return [0.0 for _ in texts_b]
    return [float(v) for v in arr_b @ arr_a]


def local_embedding_status() -> dict[str, object]:
    model_path = _model_path()
    path_exists = bool(model_path and Path(model_path).expanduser().exists())
    return {
        "enabled": _enabled(),
        "model": model_path,
        "model_path_exists": path_exists,
        "loaded": _load_model() is not None,
    }
