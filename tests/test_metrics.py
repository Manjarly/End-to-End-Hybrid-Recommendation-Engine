"""
Unit tests for evaluation metrics.
"""

import pytest
import numpy as np
from src.evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    ndcg_at_k,
    mrr_at_k,
    average_precision_at_k,
    hit_rate_at_k,
    catalog_coverage,
    novelty_at_k,
    intra_list_diversity_at_k,
)


def test_precision_and_recall_at_k():
    rec = [1, 2, 3, 4, 5]
    rel = {2, 4, 6, 8}

    p5 = precision_at_k(rec, rel, k=5)
    r5 = recall_at_k(rec, rel, k=5)

    assert p5 == 2 / 5.0
    assert r5 == 2 / 4.0


def test_ndcg_at_k():
    # Perfect ranking: relevant items at top
    rec_perfect = [1, 2, 3, 4, 5]
    rel = {1, 2}
    ndcg_perfect = ndcg_at_k(rec_perfect, rel, k=5)
    assert pytest.approx(ndcg_perfect, 0.001) == 1.0

    # Inverted ranking
    rec_delayed = [3, 4, 5, 1, 2]
    ndcg_delayed = ndcg_at_k(rec_delayed, rel, k=5)
    assert 0.0 < ndcg_delayed < 1.0


def test_mrr_and_hit_rate():
    rec = [10, 20, 30]
    rel = {20}

    assert mrr_at_k(rec, rel, k=3) == 1.0 / 2.0
    assert hit_rate_at_k(rec, rel, k=3) == 1.0

    rel_miss = {99}
    assert mrr_at_k(rec, rel_miss, k=3) == 0.0
    assert hit_rate_at_k(rec, rel_miss, k=3) == 0.0


def test_catalog_coverage():
    all_recs = {1, 2, 3, 4, 5}
    coverage = catalog_coverage(all_recs, total_catalog_items=20)
    assert coverage == 5 / 20.0


def test_diversity_and_novelty():
    genres = {
        1: "Action|Sci-Fi",
        2: "Drama|Romance",
        3: "Animation|Children's",
    }
    div = intra_list_diversity_at_k([1, 2, 3], genres, k=3)
    assert div > 0.5  # Completely different genres

    counts = {1: 100, 2: 10, 3: 1}
    nov = novelty_at_k([1, 2, 3], counts, total_users=1000, k=3)
    assert nov > 0.0
