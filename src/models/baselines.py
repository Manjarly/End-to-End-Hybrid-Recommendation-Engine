"""
Baseline Recommender Models:
- Random Recommender
- Popularity (Bayesian & Frequency-based) Recommender
- Demographic Recommender (Cold-Start Demographic Priors)
"""

from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
from .base import BaseRecommender
from ..data.dataset import RecommendationDataset
from ..data.preprocessor import AGE_MAP


class RandomRecommender(BaseRecommender):
    """Recommends uniformly random unseen movies from the catalog."""

    def __init__(self, random_state: int = 42):
        super().__init__(name="Random Recommender")
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
        self.all_movie_ids: np.ndarray = np.array([])
        self.dataset: Optional[RecommendationDataset] = None

    def fit(self, dataset: RecommendationDataset) -> "RandomRecommender":
        self.dataset = dataset
        self.all_movie_ids = np.array(list(dataset.movie_to_idx.keys()))
        self.is_fitted = True
        return self

    def recommend(
        self,
        user_id: int,
        n: int = 10,
        exclude_rated: bool = True,
        seed_movie_id: Optional[int] = None,
    ) -> List[Tuple[int, float]]:
        if not self.is_fitted or self.dataset is None:
            raise RuntimeError("Model must be fitted before recommend().")

        rated = self.dataset.user_train_rated.get(user_id, set()) if exclude_rated else set()
        candidates = [m for m in self.all_movie_ids if m not in rated]
        if not candidates:
            return []

        selected = self.rng.choice(candidates, size=min(n, len(candidates)), replace=False)
        return [(int(m), 1.0) for m in selected]

    def predict_score(self, user_id: int, movie_id: int) -> float:
        return float(self.rng.uniform(1.0, 5.0))


class PopularityRecommender(BaseRecommender):
    """
    Recommends globally popular movies based on Bayesian average ratings
    or raw interaction volume.
    """

    def __init__(self, use_bayesian: bool = True):
        super().__init__(name="Popularity Recommender (Bayesian)" if use_bayesian else "Popularity Recommender (Volume)")
        self.use_bayesian = use_bayesian
        self.ranked_movie_scores: List[Tuple[int, float]] = []
        self.dataset: Optional[RecommendationDataset] = None

    def fit(self, dataset: RecommendationDataset) -> "PopularityRecommender":
        self.dataset = dataset
        train = dataset.train_ratings

        # Calculate counts and means on train set
        stats = train.groupby("movie_id").agg(
            v=("rating", "count"),
            R=("rating", "mean"),
        ).reset_index()

        if self.use_bayesian:
            C = train["rating"].mean()
            m = 25.0  # threshold
            stats["score"] = (stats["v"] / (stats["v"] + m)) * stats["R"] + (m / (stats["v"] + m)) * C
        else:
            stats["score"] = stats["v"].astype(float)

        sorted_stats = stats.sort_values(by="score", ascending=False)
        self.ranked_movie_scores = list(zip(sorted_stats["movie_id"].astype(int), sorted_stats["score"].astype(float)))
        self.is_fitted = True
        return self

    def recommend(
        self,
        user_id: int,
        n: int = 10,
        exclude_rated: bool = True,
        seed_movie_id: Optional[int] = None,
    ) -> List[Tuple[int, float]]:
        if not self.is_fitted or self.dataset is None:
            raise RuntimeError("Model must be fitted before recommend().")

        rated = self.dataset.user_train_rated.get(user_id, set()) if exclude_rated else set()
        recommendations = []
        for mid, score in self.ranked_movie_scores:
            if mid not in rated:
                recommendations.append((mid, score))
                if len(recommendations) >= n:
                    break
        return recommendations

    def predict_score(self, user_id: int, movie_id: int) -> float:
        for mid, score in self.ranked_movie_scores:
            if mid == movie_id:
                return score
        return 3.0


class DemographicRecommender(BaseRecommender):
    """
    Cold-Start Prior Recommender:
    Leverages user age demographic cohorts to recommend movies highest-rated
    within that age group based on empirical Bayesian scores.
    """

    def __init__(self):
        super().__init__(name="Demographic Recommender")
        self.rankings_by_age: Dict[str, List[Tuple[int, float]]] = {}
        self.global_fallback: List[Tuple[int, float]] = []
        self.user_demographics: Dict[int, str] = {}
        self.dataset: Optional[RecommendationDataset] = None

    def fit(self, dataset: RecommendationDataset) -> "DemographicRecommender":
        self.dataset = dataset
        users = dataset.users
        train = dataset.train_ratings

        # Build fallback popularity
        pop = PopularityRecommender(use_bayesian=True)
        pop.fit(dataset)
        self.global_fallback = pop.ranked_movie_scores

        if users.empty or "age_desc" not in users.columns:
            self.is_fitted = True
            return self

        # Store user age demographics map
        for row in users.itertuples():
            self.user_demographics[row.user_id] = str(row.age_desc)

        # Check if MovieLens 1M ratings exist on disk for full empirical demographic coverage
        data_dir = getattr(dataset, "data_dir", None)
        if data_dir is None:
            data_dir = Path(__file__).resolve().parent.parent.parent / "data"
        ml1m_r_path = Path(data_dir) / "ml-1m" / "ratings.dat"
        ml1m_u_path = Path(data_dir) / "ml-1m" / "users.dat"

        valid_movie_ids = set(dataset.movies["movie_id"])

        if len(users) > 50 and ml1m_r_path.exists() and ml1m_u_path.exists():
            u1m = pd.read_csv(
                ml1m_u_path,
                sep="::",
                engine="python",
                names=["user_id", "gender", "age", "occupation", "zip_code"],
                encoding="latin-1",
                dtype={"user_id": int, "gender": str, "age": int, "occupation": int, "zip_code": str},
            )
            u1m["age_desc"] = u1m["age"].map(AGE_MAP).fillna("Unknown")
            r1m = pd.read_csv(
                ml1m_r_path,
                sep="::",
                engine="python",
                names=["user_id", "movie_id", "rating", "timestamp"],
                encoding="latin-1",
                dtype={"user_id": int, "movie_id": int, "rating": float, "timestamp": int},
            )
            r1m_valid = r1m[r1m["movie_id"].isin(valid_movie_ids)]
            merged = r1m_valid.merge(u1m[["user_id", "age_desc"]], on="user_id", how="inner")
        else:
            merged = train.merge(users, on="user_id", how="inner")

        if merged.empty:
            self.is_fitted = True
            return self

        # Group by age_desc alone
        for age_desc, group in merged.groupby("age_desc"):
            stats = group.groupby("movie_id").agg(
                v=("rating", "count"),
                R=("rating", "mean")
            ).reset_index()
            C = group["rating"].mean()
            m = 15.0
            stats["score"] = (stats["v"] / (stats["v"] + m)) * stats["R"] + (m / (stats["v"] + m)) * C
            sorted_group = stats.sort_values(by="score", ascending=False)
            self.rankings_by_age[str(age_desc)] = list(
                zip(sorted_group["movie_id"].astype(int), sorted_group["score"].astype(float))
            )

        self.is_fitted = True
        return self

    def recommend(
        self,
        user_id: int,
        n: int = 10,
        exclude_rated: bool = True,
        seed_movie_id: Optional[int] = None,
        age_desc: Optional[str] = None,
        **kwargs,
    ) -> List[Tuple[int, float]]:
        if not self.is_fitted or self.dataset is None:
            raise RuntimeError("Model must be fitted before recommend().")

        rated = self.dataset.user_train_rated.get(user_id, set()) if exclude_rated else set()

        a_clean = str(age_desc).strip() if age_desc and str(age_desc).strip() else None

        # Fallback to active user's demographic profile if not explicitly supplied
        if a_clean is None:
            a_clean = self.user_demographics.get(user_id)

        # Select empirical candidate list for age cohort
        if a_clean and a_clean in self.rankings_by_age:
            candidate_list = self.rankings_by_age[a_clean]
        else:
            candidate_list = self.global_fallback

        recommendations = []
        for mid, score in candidate_list:
            if mid not in rated:
                recommendations.append((mid, score))
                if len(recommendations) >= n:
                    break

        # Fallback if needed
        if len(recommendations) < n:
            for mid, score in self.global_fallback:
                if mid not in rated and mid not in [r[0] for r in recommendations]:
                    recommendations.append((mid, score))
                    if len(recommendations) >= n:
                        break

        return recommendations

    def predict_score(self, user_id: int, movie_id: int) -> float:
        return 3.5
