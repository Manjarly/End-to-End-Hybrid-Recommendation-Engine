"""
Advanced Content-Based Recommender.
Constructs rich metadata-soup representations (clean title + genres + user tags + release decade)
and computes TF-IDF similarity vectors for item-to-item and user-profile matching.
"""

from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from .base import BaseRecommender
from ..data.dataset import RecommendationDataset


class ContentBasedRecommender(BaseRecommender):
    """
    Metadata-Soup Content-Based Recommender.
    Supports:
    1. Seed-Movie Nearest Neighbors (Cosine similarity across rich tags & genres)
    2. User-Profile Content Matching (Constructs user preference vector from liked movies)
    """

    def __init__(self, max_features: int = 10000, ngram_range: Tuple[int, int] = (1, 2)):
        super().__init__(name="Content-Based (Metadata Soup)")
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.tfidf = TfidfVectorizer(
            stop_words="english",
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            sublinear_tf=True,
        )
        self.tfidf_matrix = None
        self.dataset: Optional[RecommendationDataset] = None

        # Index mappings
        self.movie_id_to_matrix_idx: Dict[int, int] = {}
        self.matrix_idx_to_movie_id: Dict[int, int] = {}

    def fit(self, dataset: RecommendationDataset) -> "ContentBasedRecommender":
        self.dataset = dataset
        movies = dataset.movies.copy().reset_index(drop=True)

        soup_texts = movies["metadata_soup"].fillna("").tolist()
        self.tfidf_matrix = self.tfidf.fit_transform(soup_texts)

        for idx, row in movies.iterrows():
            mid = int(row["movie_id"])
            self.movie_id_to_matrix_idx[mid] = idx
            self.matrix_idx_to_movie_id[idx] = mid

        self.is_fitted = True
        return self

    def get_similar_movies(self, movie_id: int, top_n: int = 10) -> List[Tuple[int, float]]:
        """Finds most similar movies based on metadata-soup cosine similarity."""
        if not self.is_fitted or self.dataset is None or self.tfidf_matrix is None:
            raise RuntimeError("Model must be fitted before get_similar_movies().")

        if movie_id not in self.movie_id_to_matrix_idx:
            return []

        idx = self.movie_id_to_matrix_idx[movie_id]
        item_vec = self.tfidf_matrix[idx]

        # Cosine similarity vector
        sim_scores = linear_kernel(item_vec, self.tfidf_matrix).flatten()

        # Sort descending, excluding the item itself
        sorted_indices = np.argsort(sim_scores)[::-1]

        results = []
        for i in sorted_indices:
            if i == idx:
                continue
            mid = self.matrix_idx_to_movie_id[i]
            score = float(sim_scores[i])
            if score > 0.01:
                results.append((mid, score))
            if len(results) >= top_n:
                break

        return results

    def recommend(
        self,
        user_id: int,
        n: int = 10,
        exclude_rated: bool = True,
        seed_movie_id: Optional[int] = None,
    ) -> List[Tuple[int, float]]:
        if not self.is_fitted or self.dataset is None or self.tfidf_matrix is None:
            raise RuntimeError("Model must be fitted before recommend().")

        rated = self.dataset.user_train_rated.get(user_id, set()) if exclude_rated else set()

        # Case 1: Seed movie provided (e.g. cold start user interested in a specific movie)
        if seed_movie_id is not None and seed_movie_id in self.movie_id_to_matrix_idx:
            sims = self.get_similar_movies(seed_movie_id, top_n=n * 3)
            filtered = [(mid, score) for mid, score in sims if mid not in rated]
            return filtered[:n]

        # Case 2: Construct user profile vector from user's high-rated movies
        user_history = self.dataset.train_ratings[
            (self.dataset.train_ratings["user_id"] == user_id) &
            (self.dataset.train_ratings["rating"] >= 3.5)
        ]

        if user_history.empty or user_id not in self.dataset.user_train_rated:
            # Fallback to top bayesian rated movies
            pop = self.dataset.movies.sort_values(by="bayesian_rating", ascending=False)
            return [(int(r.movie_id), float(r.bayesian_rating)) for r in pop.head(n).itertuples()]

        # Compute weighted average of TF-IDF vectors
        weights = []
        vec_indices = []
        for row in user_history.itertuples():
            mid = row.movie_id
            if mid in self.movie_id_to_matrix_idx:
                vec_indices.append(self.movie_id_to_matrix_idx[mid])
                weights.append(row.rating - 2.5)  # weight higher-rated movies more

        if not vec_indices:
            pop = self.dataset.movies.sort_values(by="bayesian_rating", ascending=False)
            return [(int(r.movie_id), float(r.bayesian_rating)) for r in pop.head(n).itertuples()]

        sub_matrix = self.tfidf_matrix[vec_indices]
        w = np.array(weights).reshape(1, -1)
        w = w / np.sum(w)  # normalize weights

        # Profile vector: 1 x vocab_size
        user_profile = w.dot(sub_matrix.toarray())

        # Cosine similarity against all catalog movies
        profile_sims = linear_kernel(user_profile, self.tfidf_matrix).flatten()
        sorted_indices = np.argsort(profile_sims)[::-1]

        recommendations = []
        for i in sorted_indices:
            mid = self.matrix_idx_to_movie_id[i]
            if mid not in rated:
                recommendations.append((mid, float(profile_sims[i])))
                if len(recommendations) >= n:
                    break

        return recommendations

    def predict_score(self, user_id: int, movie_id: int) -> float:
        if not self.is_fitted or self.tfidf_matrix is None or movie_id not in self.movie_id_to_matrix_idx:
            return 3.0

        recs = dict(self.recommend(user_id, n=100, exclude_rated=False))
        return recs.get(movie_id, 3.0)
