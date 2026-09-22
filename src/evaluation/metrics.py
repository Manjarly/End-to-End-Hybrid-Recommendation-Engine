"""
Comprehensive Evaluation Metrics for Recommender Systems.
Ranking Accuracy:
- Precision@K, Recall@K, NDCG@K, MAP@K, MRR@K, HitRate@K
Beyond-Accuracy:
- Catalog Coverage, Novelty (Surprisal), Intra-List Diversity
"""

from typing import Dict, List, Sequence, Set, Tuple
import numpy as np


def precision_at_k(recommended: Sequence[int], relevant: Set[int], k: int = 10) -> float:
    """Calculates Precision@K: Fraction of top-K recommended items that are relevant."""
    if k <= 0 or not relevant:
        return 0.0
    top_k = recommended[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / float(k)


def recall_at_k(recommended: Sequence[int], relevant: Set[int], k: int = 10) -> float:
    """Calculates Recall@K: Fraction of relevant items retrieved in top-K."""
    if not relevant:
        return 0.0
    top_k = recommended[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / float(len(relevant))


def hit_rate_at_k(recommended: Sequence[int], relevant: Set[int], k: int = 10) -> float:
    """Calculates Hit Rate@K: 1 if at least one relevant item in top-K, 0 otherwise."""
    top_k = recommended[:k]
    for item in top_k:
        if item in relevant:
            return 1.0
    return 0.0


def mrr_at_k(recommended: Sequence[int], relevant: Set[int], k: int = 10) -> float:
    """Calculates Mean Reciprocal Rank@K: 1 / rank of first relevant item in top-K."""
    top_k = recommended[:k]
    for rank, item in enumerate(top_k, 1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def average_precision_at_k(recommended: Sequence[int], relevant: Set[int], k: int = 10) -> float:
    """Calculates Average Precision@K (AP@K) for a single user."""
    if not relevant:
        return 0.0
    top_k = recommended[:k]
    score = 0.0
    hits = 0
    for i, item in enumerate(top_k):
        if item in relevant:
            hits += 1
            score += hits / (i + 1.0)
    return score / min(len(relevant), k)


def ndcg_at_k(recommended: Sequence[int], relevant: Set[int], k: int = 10) -> float:
    """
    Calculates Normalized Discounted Cumulative Gain@K (NDCG@K) with binary relevance.
    DCG = sum_{i=1}^K (rel_i / log2(i + 1))
    """
    if not relevant:
        return 0.0
    top_k = recommended[:k]
    dcg = 0.0
    for i, item in enumerate(top_k):
        if item in relevant:
            dcg += 1.0 / np.log2(i + 2)  # i+2 because i starts at 0

    # Ideal DCG
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return (dcg / idcg) if idcg > 0 else 0.0


def catalog_coverage(all_recommended_items: Set[int], total_catalog_items: int) -> float:
    """Calculates Catalog Coverage: % of catalog items recommended at least once."""
    if total_catalog_items <= 0:
        return 0.0
    return len(all_recommended_items) / float(total_catalog_items)


def novelty_at_k(
    recommended: Sequence[int],
    item_interaction_counts: Dict[int, int],
    total_users: int,
    k: int = 10,
) -> float:
    """
    Calculates Novelty (Self-Information / Surprisal) of recommendations:
    Novelty(u) = (1 / K) * sum_{i in TopK} -log2(P(i))
    Higher values indicate recommendations of niche items/hidden gems rather than obvious hits.
    """
    top_k = recommended[:k]
    if not top_k or total_users <= 0:
        return 0.0

    self_info_sum = 0.0
    for mid in top_k:
        count = item_interaction_counts.get(mid, 1)
        p_i = max(count / float(total_users), 1e-6)
        self_info_sum += -np.log2(p_i)

    return self_info_sum / float(len(top_k))


def intra_list_diversity_at_k(
    recommended: Sequence[int],
    item_genres: Dict[int, str],
    k: int = 10,
) -> float:
    """
    Calculates Intra-List Diversity using pairwise Jaccard distance between item genres.
    Distance = 1 - (Intersection / Union)
    """
    top_k = recommended[:k]
    if len(top_k) < 2:
        return 0.0

    genre_sets = [
        set(item_genres.get(mid, "").split("|")) - {"", "unknown"}
        for mid in top_k
    ]

    distances = []
    for i in range(len(genre_sets)):
        for j in range(i + 1, len(genre_sets)):
            s1 = genre_sets[i]
            s2 = genre_sets[j]
            union = len(s1 | s2)
            if union == 0:
                dist = 0.5
            else:
                jaccard = len(s1 & s2) / union
                dist = 1.0 - jaccard
            distances.append(dist)

    return float(np.mean(distances)) if distances else 0.0
