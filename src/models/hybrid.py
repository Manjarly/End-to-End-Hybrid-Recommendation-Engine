"""
State-of-the-Art Hybrid Recommender System.
Integrates:
1. Reciprocal Rank Fusion (RRF)
2. Weighted Linear Score Blending with Interactive Alpha Tuning
3. Dynamic Cold-Start & Demographic Switching Policy
4. Diversity-Aware Maximal Marginal Relevance (MMR) Re-Ranking
"""

from typing import Dict, Iterable, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
from .base import BaseRecommender
from .collaborative import BiasedSVDRecommender
from .content_based import ContentBasedRecommender
from .baselines import PopularityRecommender, DemographicRecommender
from ..data.dataset import RecommendationDataset


class HybridRecommender(BaseRecommender):
    """
    Advanced Hybrid Recommendation Engine combining Collaborative Filtering,
    Content-Based Filtering, and Demographic Priors.
    """

    def __init__(
        self,
        cf_model: Optional[BiasedSVDRecommender] = None,
        cb_model: Optional[ContentBasedRecommender] = None,
        alpha: float = 0.65,  # 0.65 CF, 0.35 CB
        fusion_strategy: str = "rrf",  # 'rrf' or 'weighted'
        rrf_k: int = 60,
    ):
        super().__init__(name=f"Hybrid Engine ({fusion_strategy.upper()})")
        self.cf_model = cf_model if cf_model is not None else BiasedSVDRecommender(n_components=40)
        self.cb_model = cb_model if cb_model is not None else ContentBasedRecommender()
        self.popularity_model = PopularityRecommender(use_bayesian=True)
        self.demographic_model = DemographicRecommender()

        self.alpha = alpha  # Weight for CF vs CB in weighted fusion
        self.fusion_strategy = fusion_strategy
        self.rrf_k = rrf_k
        self.dataset: Optional[RecommendationDataset] = None

    def fit(self, dataset: RecommendationDataset) -> "HybridRecommender":
        self.dataset = dataset
        print(f"[HybridRecommender] Fitting Collaborative Filtering model ({self.cf_model.name})...")
        self.cf_model.fit(dataset)

        print(f"[HybridRecommender] Fitting Content-Based model ({self.cb_model.name})...")
        self.cb_model.fit(dataset)

        print("[HybridRecommender] Fitting Baseline & Demographic models...")
        self.popularity_model.fit(dataset)
        self.demographic_model.fit(dataset)

        self.is_fitted = True
        print("[HybridRecommender] Hybrid Engine successfully fitted.")
        return self

    def recommend(
        self,
        user_id: int,
        n: int = 10,
        exclude_rated: bool = True,
        seed_movie_id: Optional[int] = None,
        alpha: Optional[float] = None,
        strategy: Optional[str] = None,
        age_desc: Optional[str] = None,
        apply_diversity: bool = False,
        exclude_movie_ids: Optional[Iterable[int]] = None,
        **kwargs,
    ) -> List[Tuple[int, float]]:
        """
        Generates hybrid recommendations with dynamic cold-start routing.
        """
        if not self.is_fitted or self.dataset is None:
            raise RuntimeError("Model must be fitted before recommend().")

        alpha_val = self.alpha if alpha is None else alpha
        strat = self.fusion_strategy if strategy is None else strategy

        rated = self.dataset.user_train_rated.get(user_id, set()) if exclude_rated else set()
        user_rating_count = len(rated)

        # Build comprehensive exclusion set (rated movies, seed movie, and journey trail history)
        exclude_set: Set[int] = set(rated)
        if seed_movie_id is not None:
            exclude_set.add(seed_movie_id)
        if exclude_movie_ids:
            exclude_set.update(int(x) for x in exclude_movie_ids)

        # -------------------------------------------------------------
        # 1. Cold-Start Handling (User has 0 ratings)
        # -------------------------------------------------------------
        if user_rating_count == 0:
            return self._handle_cold_start(
                user_id=user_id,
                n=n,
                seed_movie_id=seed_movie_id,
                age_desc=age_desc,
                exclude_movie_ids=exclude_set,
            )

        # -------------------------------------------------------------
        # 2. Warm User (1-4 ratings): Dynamic Alpha Interpolation
        # -------------------------------------------------------------
        if user_rating_count < 5:
            # Gradually shift from CB towards CF as interactions accumulate
            effective_alpha = alpha_val * (user_rating_count / 5.0)
        else:
            effective_alpha = alpha_val

        # When a reference seed film is specified for exploration, prioritize content kinship
        # with the seed while retaining collaborative filtering as a personalized ranking signal
        if seed_movie_id is not None:
            effective_alpha = min(effective_alpha * 0.35, 0.25)

        # -------------------------------------------------------------
        # 3. Retrieve Candidate Pool from Models
        # -------------------------------------------------------------
        candidate_k = max(n * 4, 60)
        cf_recs = self.cf_model.recommend(user_id, n=candidate_k, exclude_rated=exclude_rated)
        cb_recs = self.cb_model.recommend(user_id, n=candidate_k, exclude_rated=exclude_rated, seed_movie_id=seed_movie_id)

        # Retrieve demographic candidates if age cohort specified
        has_demo = bool(age_desc and str(age_desc).strip())
        demo_recs = []
        if has_demo:
            demo_recs = self.demographic_model.recommend(
                user_id=user_id,
                n=candidate_k,
                exclude_rated=exclude_rated,
                age_desc=age_desc,
            )

        # -------------------------------------------------------------
        # 4. Fusion Strategy
        # -------------------------------------------------------------
        if strat == "weighted":
            final_scores = self._weighted_score_fusion(cf_recs, cb_recs, effective_alpha, demo_recs=demo_recs)
        else:
            # Reciprocal Rank Fusion (RRF)
            final_scores = self._reciprocal_rank_fusion(cf_recs, cb_recs, effective_alpha, demo_recs=demo_recs)

        # Filter already rated, seed movie, and any discovery graph history
        ranked_items = [
            (mid, score) for mid, score in final_scores
            if mid not in exclude_set
        ]

        if apply_diversity:
            ranked_items = self._apply_mmr_diversity(ranked_items, n)
        else:
            ranked_items = ranked_items[:n]

        return ranked_items

    def _handle_cold_start(
        self,
        user_id: int,
        n: int,
        seed_movie_id: Optional[int] = None,
        age_desc: Optional[str] = None,
        exclude_movie_ids: Optional[Set[int]] = None,
        **kwargs,
    ) -> List[Tuple[int, float]]:
        """Handles new users who have not rated any movies yet."""
        has_demo = bool(age_desc and str(age_desc).strip())
        excl: Set[int] = set(exclude_movie_ids) if exclude_movie_ids else set()
        if seed_movie_id is not None:
            excl.add(seed_movie_id)

        # Case A: User picked a reference film
        if seed_movie_id is not None:
            cb_sims = self.cb_model.get_similar_movies(seed_movie_id, top_n=n * 4)
            cb_sims = [(mid, s) for mid, s in cb_sims if mid not in excl]
            if has_demo:
                demo_recs = self.demographic_model.recommend(user_id, n=n * 4, age_desc=age_desc)
                if demo_recs:
                    fused = self._reciprocal_rank_fusion(cf_recs=[], cb_recs=cb_sims, alpha=0.0, demo_recs=demo_recs)
                    return [(mid, s) for mid, s in fused if mid not in excl][:n]
            return [(mid, float(s)) for mid, s in cb_sims if mid not in excl][:n]

        # Case B: Demographic profile selected (Age Group)
        if has_demo or user_id in self.demographic_model.user_demographics:
            demo_recs = self.demographic_model.recommend(user_id, n=n * 2, age_desc=age_desc)
            if demo_recs:
                return [(mid, s) for mid, s in demo_recs if mid not in excl][:n]

        # Case C: Global Bayesian Popularity
        pop_recs = self.popularity_model.recommend(user_id, n=n * 2)
        return [(mid, s) for mid, s in pop_recs if mid not in excl][:n]

    def _reciprocal_rank_fusion(
        self,
        cf_recs: List[Tuple[int, float]],
        cb_recs: List[Tuple[int, float]],
        alpha: float,
        demo_recs: Optional[List[Tuple[int, float]]] = None,
    ) -> List[Tuple[int, float]]:
        """Combines rankings using Reciprocal Rank Fusion (RRF)."""
        rrf_scores: Dict[int, float] = {}

        if demo_recs:
            w_cf = alpha * 0.75
            w_cb = (1.0 - alpha) * 0.75
            w_demo = 0.25
        else:
            w_cf = alpha
            w_cb = 1.0 - alpha
            w_demo = 0.0

        for rank, (mid, _) in enumerate(cf_recs, 1):
            rrf_scores[mid] = rrf_scores.get(mid, 0.0) + w_cf / (self.rrf_k + rank)

        for rank, (mid, _) in enumerate(cb_recs, 1):
            rrf_scores[mid] = rrf_scores.get(mid, 0.0) + w_cb / (self.rrf_k + rank)

        if demo_recs:
            for rank, (mid, _) in enumerate(demo_recs, 1):
                rrf_scores[mid] = rrf_scores.get(mid, 0.0) + w_demo / (self.rrf_k + rank)

        sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_items

    def _weighted_score_fusion(
        self,
        cf_recs: List[Tuple[int, float]],
        cb_recs: List[Tuple[int, float]],
        alpha: float,
        demo_recs: Optional[List[Tuple[int, float]]] = None,
    ) -> List[Tuple[int, float]]:
        """Combines normalized scores via linear weighting."""
        def min_max_norm(recs: List[Tuple[int, float]]) -> Dict[int, float]:
            if not recs:
                return {}
            scores = [s for _, s in recs]
            s_min, s_max = min(scores), max(scores)
            denom = s_max - s_min if s_max > s_min else 1.0
            return {mid: (s - s_min) / denom for mid, s in recs}

        norm_cf = min_max_norm(cf_recs)
        norm_cb = min_max_norm(cb_recs)
        norm_demo = min_max_norm(demo_recs) if demo_recs else {}

        all_ids = set(norm_cf.keys()) | set(norm_cb.keys()) | set(norm_demo.keys())
        blended: Dict[int, float] = {}

        if norm_demo:
            w_cf = alpha * 0.75
            w_cb = (1.0 - alpha) * 0.75
            w_demo = 0.25
        else:
            w_cf = alpha
            w_cb = 1.0 - alpha
            w_demo = 0.0

        for mid in all_ids:
            score_cf = norm_cf.get(mid, 0.0)
            score_cb = norm_cb.get(mid, 0.0)
            score_demo = norm_demo.get(mid, 0.0)
            blended[mid] = w_cf * score_cf + w_cb * score_cb + w_demo * score_demo

        return sorted(blended.items(), key=lambda x: x[1], reverse=True)

    def _apply_mmr_diversity(
        self,
        ranked_candidates: List[Tuple[int, float]],
        top_n: int,
        diversity_lambda: float = 0.7,
    ) -> List[Tuple[int, float]]:
        """
        Maximal Marginal Relevance (MMR) re-ranking.
        Balances recommendation relevance against diversity of movie genres.
        """
        if len(ranked_candidates) <= top_n or self.dataset is None:
            return ranked_candidates[:top_n]

        selected: List[Tuple[int, float]] = [ranked_candidates[0]]
        candidates = ranked_candidates[1:]

        # Precompute genre sets
        item_genres = {
            mid: set(self.dataset.movie_id_to_genres.get(mid, "").split("|"))
            for mid, _ in ranked_candidates
        }

        while len(selected) < top_n and candidates:
            best_idx = -1
            best_mmr_score = -float("inf")

            for idx, (mid, rel_score) in enumerate(candidates):
                g_mid = item_genres.get(mid, set())

                # Compute maximum genre overlap with already selected items (Jaccard similarity)
                max_sim = 0.0
                for sel_mid, _ in selected:
                    g_sel = item_genres.get(sel_mid, set())
                    union = len(g_mid | g_sel)
                    if union > 0:
                        jaccard = len(g_mid & g_sel) / union
                        if jaccard > max_sim:
                            max_sim = jaccard

                mmr_score = diversity_lambda * rel_score - (1.0 - diversity_lambda) * max_sim
                if mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_idx = idx

            selected.append(candidates.pop(best_idx))

        return selected

    def predict_score(self, user_id: int, movie_id: int) -> float:
        cf_score = self.cf_model.predict_score(user_id, movie_id)
        cb_score = self.cb_model.predict_score(user_id, movie_id)
        return float(self.alpha * cf_score + (1.0 - self.alpha) * cb_score)
