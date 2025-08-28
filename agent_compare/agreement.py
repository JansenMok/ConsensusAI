from typing import List, Dict, Any
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
from sentence_transformers import SentenceTransformer
from .types import Decision  # use with __init__.py only

def _cluster_precomputed(D, distance_threshold: float):
    """
    Create an AgglomerativeClustering that accepts a precomputed distance matrix.
    Works on both new (metric=) and old (affinity=) sklearn versions.
    """
    try:
        # New API (sklearn >= 1.4)
        return AgglomerativeClustering(
            metric="precomputed",
            linkage="average",
            distance_threshold=distance_threshold,
            n_clusters=None,
        )
    except TypeError:
        # Old API (sklearn < 1.4)
        return AgglomerativeClustering(
            affinity="precomputed",
            linkage="average",
            distance_threshold=distance_threshold,
            n_clusters=None,
        )

# Sentence-BERT model
def _embed(texts: List[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    model = SentenceTransformer(model_name)
    return model.encode(texts, normalize_embeddings=True)

def agreement_decision(
    texts: List[str],
    *,
    min_core_fraction: float = 0.6,
    distance_threshold: float = 0.25,   # 1 - cosine; ~0.75 cosine boundary
    mean_cos_threshold: float = 0.82,
    min_cos_threshold: float = 0.70,
    require_silhouette: float = 0.20,
    embed_model: str = "all-MiniLM-L6-v2",
    clust = _cluster_precomputed(D, distance_threshold),
    labels = clust.fit_predict(D),
) -> Decision:
    assert len(texts) >= 2, "need at least 2 texts"

    E = _embed(texts, embed_model)
    S = cosine_similarity(E)
    n = len(texts)
    mean_cos = float((S.sum() - n) / (n*(n-1)))
    # ignore diagonal; add 2 trick to avoid selecting it as min
    min_cos = float((S + np.eye(n)*2).min())

    # clustering in distance space
    D = 1.0 - S
    clust = AgglomerativeClustering(
        affinity="precomputed", linkage="average",
        distance_threshold=distance_threshold, n_clusters=None
    )
    labels = clust.fit_predict(D)
    unique, counts = np.unique(labels, return_counts=True)
    core_label = int(unique[np.argmax(counts)])
    core_idx = [i for i, lab in enumerate(labels) if lab == core_label]
    core_frac = len(core_idx) / n

    # silhouette on cosine distance → silhouette needs similarity or distance; we pass distance
    sil = silhouette_score(D, labels, metric="precomputed") if len(unique) > 1 else 0.0

    agreement = (
        mean_cos >= mean_cos_threshold
        and min_cos >= min_cos_threshold
        and core_frac >= min_core_fraction
        and sil >= require_silhouette
    )

    why = {
        "cluster_sizes": {int(u): int(c) for u, c in zip(unique.tolist(), counts.tolist())},
        "core_fraction": core_frac,
        "silhouette": float(sil),
        "thresholds": {
            "mean_cos": mean_cos_threshold,
            "min_cos": min_cos_threshold,
            "core_fraction": min_core_fraction,
            "silhouette": require_silhouette,
        },
    }

    return Decision(
        agreement=bool(agreement),
        mean_cos=mean_cos,
        min_cos=min_cos,
        core_idx=core_idx,
        pairwise=S.tolist(),
        labels=labels.astype(int).tolist(),
        why=why,
    )
