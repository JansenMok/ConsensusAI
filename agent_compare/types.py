from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class ProviderReply:
    provider: str
    model: str
    text: str
    latency_ms: Optional[int] = None
    refusal: bool = False
    meta: Optional[Dict[str, Any]] = None

@dataclass
class Decision:
    agreement: bool
    mean_cos: float
    min_cos: float
    core_idx: list[int]           # indexes of replies considered the “core cluster”
    pairwise: list[list[float]]   # cosine similarity matrix
    labels: list[int]             # cluster label per reply
    why: dict                     # extra diagnostics (cluster sizes, silhouette, etc.)
