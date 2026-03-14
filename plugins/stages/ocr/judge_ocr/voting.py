"""
Voting strategies for Judge OCR.

Each strategy receives a list of candidates from different engines for
the same spatial region and returns the single best (text, confidence) pair.

A candidate is a dict: {"text": str, "confidence": float, "engine": str}
"""

from __future__ import annotations

import difflib
from collections import defaultdict


def text_similarity(a: str, b: str) -> float:
    """Normalized similarity ratio between two strings (0.0 – 1.0)."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def _cluster_by_text(candidates: list[dict], similarity_threshold: float = 0.7) -> list[list[dict]]:
    """
    Group candidates whose texts are similar into clusters.

    Uses single-linkage: a candidate joins the first cluster where it
    matches any existing member above *similarity_threshold*.
    """
    clusters: list[list[dict]] = []
    for cand in candidates:
        placed = False
        for cluster in clusters:
            for member in cluster:
                if text_similarity(cand["text"], member["text"]) >= similarity_threshold:
                    cluster.append(cand)
                    placed = True
                    break
            if placed:
                break
        if not placed:
            clusters.append([cand])
    return clusters


def majority_vote(candidates: list[dict]) -> tuple[str, float]:
    """
    Pick the text cluster with the most votes, then return the
    highest-confidence result from that cluster.
    """
    if not candidates:
        return ("", 0.0)
    if len(candidates) == 1:
        return (candidates[0]["text"], candidates[0]["confidence"])

    clusters = _cluster_by_text(candidates)
    best_cluster = max(clusters, key=len)
    winner = max(best_cluster, key=lambda c: c["confidence"])
    return (winner["text"], winner["confidence"])


def weighted_confidence(candidates: list[dict]) -> tuple[str, float]:
    """
    Sum confidence scores per text cluster; pick the cluster with the
    highest total, then return its highest-confidence member.
    """
    if not candidates:
        return ("", 0.0)
    if len(candidates) == 1:
        return (candidates[0]["text"], candidates[0]["confidence"])

    clusters = _cluster_by_text(candidates)
    best_cluster = max(clusters, key=lambda cl: sum(c["confidence"] for c in cl))
    winner = max(best_cluster, key=lambda c: c["confidence"])
    return (winner["text"], winner["confidence"])


def quorum(candidates: list[dict], min_count: int = 2) -> tuple[str, float] | None:
    """
    Accept a region only if at least *min_count* engines detected text in
    it.  Among those, return the highest-confidence result.

    Returns ``None`` if quorum is not met (caller should discard the region).
    """
    if len(candidates) < min_count:
        return None
    winner = max(candidates, key=lambda c: c["confidence"])
    return (winner["text"], winner["confidence"])


def best_confidence(candidates: list[dict]) -> tuple[str, float]:
    """Pick the single candidate with the highest confidence."""
    if not candidates:
        return ("", 0.0)
    winner = max(candidates, key=lambda c: c["confidence"])
    return (winner["text"], winner["confidence"])


STRATEGIES: dict[str, object] = {
    "majority_vote": majority_vote,
    "weighted_confidence": weighted_confidence,
    "quorum": quorum,
    "best_confidence": best_confidence,
}
