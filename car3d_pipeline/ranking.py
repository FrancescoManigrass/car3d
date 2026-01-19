from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

from .config import RankingConfig


@dataclass(frozen=True)
class ViewScore:
    view_name: str
    clip_score: float
    silhouette_iou: float
    hardpoint_penalty: float

    def total(self, config: RankingConfig) -> float:
        return (
            config.clip_weight * self.clip_score
            + config.silhouette_weight * self.silhouette_iou
            - config.hardpoint_penalty_weight * self.hardpoint_penalty
        )


def compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return float(intersection / union) if union > 0 else 0.0


def load_mask(path: Path, threshold: int = 127) -> np.ndarray:
    image = Image.open(path).convert("L")
    return np.array(image) > threshold


def aggregate_scores(scores: Iterable[ViewScore], config: RankingConfig) -> float:
    total = 0.0
    weight_sum = 0.0
    for score in scores:
        weight = config.view_weights.get(score.view_name, 1.0)
        total += score.total(config) * weight
        weight_sum += weight
    return total / weight_sum if weight_sum > 0 else 0.0


def placeholder_clip_score() -> float:
    """Return a placeholder CLIP score when CLIP is not wired in."""
    return 0.0
