"""
Collaborative Filtering Algorithms:
1. Biased Matrix Factorization (SVD with User & Item Biases)
2. Item-Item Collaborative Filtering (Cosine with Significance Shrinkage)
"""

from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from .base import BaseRecommender
from ..data.dataset import RecommendationDataset


class BiasedSVDRecommender(BaseRecommender):
    """
    Biased Matrix Factorization Recommender.
    Decomposes rating: r_hat(u, i) = mu + b_u + b_i + P_u * Q_i^T
    Solves the fatal zero-fill distortion of naive SVD by centering around
    global, user, and item rating biases.
    """

    def __init__(self, n_components: int = 50, reg_user: float = 10.0, reg_item: float = 10.0, random_state: int = 42):
        super().__init__(name=f"Biased SVD (k={n_components})")
        self.n_components = n_components
        self.reg_user = reg_user
        self.reg_item = reg_item
        self.random_state = random_state

        self.global_mean: float = 0.0
        self.user_bias: np.ndarray = np.array([])
        self.item_bias: np.ndarray = np.array([])
        self.user_factors: np.ndarray = np.array([])
        self.item_factors: np.ndarray = np.array([])

        self.svd: Optional[TruncatedSVD] = None
        self.dataset: Optional[RecommendationDataset] = None

    def fit(self, dataset: RecommendationDataset) -> "BiasedSVDRecommender":
        self.dataset = dataset
        train = dataset.train_ratings

        n_users = len(dataset.user_to_idx)
        n_movies = len(dataset.movie_to_idx)

        # 1. Global Mean
        self.global_mean = float(train["rating"].mean())

        # 2. User Biases with L2 Regularization (Shrinkage)
        u_counts = np.zeros(n_users, dtype=np.float32)
        u_sums = np.zeros(n_users, dtype=np.float32)
        for row in train.itertuples():
            uidx = dataset.user_to_idx[row.user_id]
            u_counts[uidx] += 1
            u_sums[uidx] += (row.rating - self.global_mean)

        self.user_bias = u_sums / (u_counts + self.reg_user)

        # 3. Item Biases with L2 Regularization
        i_counts = np.zeros(n_movies, dtype=np.float32)
        i_sums = np.zeros(n_movies, dtype=np.float32)
        for row in train.itertuples():
            uidx = dataset.user_to_idx[row.user_id]
            midx = dataset.movie_to_idx[row.movie_id]
            i_counts[midx] += 1
            i_sums[midx] += (row.rating - self.global_mean - self.user_bias[uidx])

        self.item_bias = i_sums / (i_counts + self.reg_item)

        # 4. Residual Matrix for Matrix Factorization
        rows = []
        cols = []
        residuals = []
        for row in train.itertuples():
            uidx = dataset.user_to_idx[row.user_id]
            midx = dataset.movie_to_idx[row.movie_id]
            residual = row.rating - (self.global_mean + self.user_bias[uidx] + self.item_bias[midx])
            rows.append(uidx)
            cols.append(midx)
            residuals.append(residual)

        res_matrix = csr_matrix((residuals, (rows, cols)), shape=(n_users, n_movies))

        # 5. Latent Factor SVD Decomposition
        k = min(self.n_components, min(n_users, n_movies) - 1)
        self.svd = TruncatedSVD(n_components=k, random_state=self.random_state)
        self.user_factors = self.svd.fit_transform(res_matrix)  # (n_users, k)
        self.item_factors = self.svd.components_.T              # (n_movies, k)

        self.is_fitted = True
        return self

    def predict_score(self, user_id: int, movie_id: int) -> float:
        if not self.is_fitted or self.dataset is None:
            raise RuntimeError("Model must be fitted before predict_score().")

        if user_id not in self.dataset.user_to_idx or movie_id not in self.dataset.movie_to_idx:
            # Fallback for unknown user or item
            item_b = self.item_bias[self.dataset.movie_to_idx[movie_id]] if movie_id in self.dataset.movie_to_idx else 0.0
            return float(np.clip(self.global_mean + item_b, 1.0, 5.0))

        uidx = self.dataset.user_to_idx[user_id]
        midx = self.dataset.movie_to_idx[movie_id]

        pred = (
            self.global_mean
            + self.user_bias[uidx]
            + self.item_bias[midx]
            + np.dot(self.user_factors[uidx], self.item_factors[midx])
        )
        return float(np.clip(pred, 1.0, 5.0))

    def recommend(
        self,
        user_id: int,
        n: int = 10,
        exclude_rated: bool = True,
        seed_movie_id: Optional[int] = None,
    ) -> List[Tuple[int, float]]:
        if not self.is_fitted or self.dataset is None:
            raise RuntimeError("Model must be fitted before recommend().")

        if user_id not in self.dataset.user_to_idx:
            # Cold start fallback
            pop = self.dataset.movies.sort_values(by="bayesian_rating", ascending=False)
            return [(int(r.movie_id), float(r.bayesian_rating)) for r in pop.head(n).itertuples()]

        uidx = self.dataset.user_to_idx[user_id]
        u_factor = self.user_factors[uidx]
        u_b = self.user_bias[uidx]

        # Fast vector prediction across all items
        latent_scores = np.dot(self.item_factors, u_factor)
        predicted_ratings = self.global_mean + u_b + self.item_bias + latent_scores

        rated = self.dataset.user_train_rated.get(user_id, set()) if exclude_rated else set()

        # Build scored recommendations
        results = []
        for mid, midx in self.dataset.movie_to_idx.items():
            if mid not in rated:
                results.append((mid, float(predicted_ratings[midx])))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:n]


class ItemItemCFRecommender(BaseRecommender):
    """
    Item-Item Collaborative Filtering with Adjusted Cosine & Significance Shrinkage.
    Finds items rated similarly by common users.
    """

    def __init__(self, top_k_neighbors: int = 30, shrinkage: float = 25.0):
        super().__init__(name=f"Item-Item CF (k={top_k_neighbors})")
        self.top_k_neighbors = top_k_neighbors
        self.shrinkage = shrinkage
        self.dataset: Optional[RecommendationDataset] = None

        # item similarity lookup: item_id -> list of (neighbor_item_id, similarity)
        self.item_similarities: Dict[int, List[Tuple[int, float]]] = {}
        self.item_mean_ratings: Dict[int, float] = {}

    def fit(self, dataset: RecommendationDataset) -> "ItemItemCFRecommender":
        self.dataset = dataset
        train = dataset.train_ratings

        # Compute item mean ratings
        item_stats = train.groupby("movie_id")["rating"].agg(["mean", "count"]).to_dict("index")
        self.item_mean_ratings = {mid: stats["mean"] for mid, stats in item_stats.items()}

        # Build user-item rating dict
        user_ratings: Dict[int, Dict[int, float]] = {}
        item_users: Dict[int, Dict[int, float]] = {}

        for row in train.itertuples():
            uid = row.user_id
            mid = row.movie_id
            r = row.rating
            if uid not in user_ratings:
                user_ratings[uid] = {}
            user_ratings[uid][mid] = r

            if mid not in item_users:
                item_users[mid] = {}
            item_users[mid][uid] = r

        # Compute top item-item similarities for active catalog items
        active_items = [mid for mid, stats in item_stats.items() if stats["count"] >= 5]

        for i, mid_i in enumerate(active_items):
            users_i = item_users[mid_i]
            sims = []
            for mid_j in active_items:
                if mid_i == mid_j:
                    continue
                users_j = item_users[mid_j]
                common_users = set(users_i.keys()) & set(users_j.keys())
                if len(common_users) < 3:
                    continue

                # Cosine similarity on ratings
                vec_i = [users_i[u] for u in common_users]
                vec_j = [users_j[u] for u in common_users]
                dot = sum(a * b for a, b in zip(vec_i, vec_j))
                norm_i = np.sqrt(sum(a * a for a in vec_i))
                norm_j = np.sqrt(sum(b * b for b in vec_j))
                if norm_i > 0 and norm_j > 0:
                    raw_sim = dot / (norm_i * norm_j)
                    # Significance shrinkage
                    shrunk_sim = raw_sim * (min(len(common_users), self.shrinkage) / self.shrinkage)
                    sims.append((mid_j, float(shrunk_sim)))

            sims.sort(key=lambda x: x[1], reverse=True)
            self.item_similarities[mid_i] = sims[:self.top_k_neighbors]

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

        if seed_movie_id is not None:
            # Recommend directly similar items to seed
            sims = self.item_similarities.get(seed_movie_id, [])
            return [(mid, sim) for mid, sim in sims if mid not in rated][:n]

        # Aggregate neighbor scores across user's rated items
        scores: Dict[int, float] = {}
        weights: Dict[int, float] = {}

        user_ratings = self.dataset.train_ratings[self.dataset.train_ratings["user_id"] == user_id]
        if user_ratings.empty:
            # Fallback to bayesian popularity
            pop = self.dataset.movies.sort_values(by="bayesian_rating", ascending=False)
            return [(int(r.movie_id), float(r.bayesian_rating)) for r in pop.head(n).itertuples()]

        for row in user_ratings.itertuples():
            mid = row.movie_id
            r = row.rating
            for neighbor_id, sim in self.item_similarities.get(mid, []):
                if neighbor_id not in rated:
                    scores[neighbor_id] = scores.get(neighbor_id, 0.0) + sim * r
                    weights[neighbor_id] = weights.get(neighbor_id, 0.0) + abs(sim)

        results = []
        for mid, score_sum in scores.items():
            w = weights.get(mid, 1.0)
            if w > 0:
                results.append((mid, score_sum / w))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:n]

    def predict_score(self, user_id: int, movie_id: int) -> float:
        return self.item_mean_ratings.get(movie_id, 3.5)
