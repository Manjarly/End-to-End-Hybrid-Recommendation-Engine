"""
Recommendation dataset container and train/test partitioner.
Optimized for high-performance retrieval and evaluation.
"""

from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


class RecommendationDataset:
    """Encapsulates movies, ratings, users, train/test splits, and sparse matrices."""

    def __init__(
        self,
        movies_df: pd.DataFrame,
        ratings_df: pd.DataFrame,
        users_df: Optional[pd.DataFrame] = None,
        test_size: float = 0.20,
        min_relevance_rating: float = 4.0,
        random_state: int = 42,
    ):
        self.movies = movies_df.copy()
        self.ratings = ratings_df.copy()
        self.users = users_df.copy() if users_df is not None else pd.DataFrame()
        self.test_size = test_size
        self.min_relevance_rating = min_relevance_rating
        self.random_state = random_state

        # Index lookups
        self.movie_id_to_title = dict(zip(self.movies["movie_id"], self.movies["title"]))
        self.title_to_movie_id = dict(zip(self.movies["title"], self.movies["movie_id"]))
        self.movie_id_to_genres = dict(zip(self.movies["movie_id"], self.movies["genres"]))

        # Build train/test splits
        self.train_ratings: pd.DataFrame = pd.DataFrame()
        self.test_ratings: pd.DataFrame = pd.DataFrame()
        self.user_train_rated: Dict[int, Set[int]] = defaultdict(set)
        self.user_test_relevant: Dict[int, List[int]] = defaultdict(list)
        self.user_test_all: Dict[int, List[int]] = defaultdict(list)

        # Mappings for sparse matrix
        self.user_to_idx: Dict[int, int] = {}
        self.idx_to_user: Dict[int, int] = {}
        self.movie_to_idx: Dict[int, int] = {}
        self.idx_to_movie: Dict[int, int] = {}

        self._split_data()

    def _split_data(self) -> None:
        """Splits ratings into train and test sets using user-stratified sampling."""
        rng = np.random.RandomState(self.random_state)

        # To avoid data leakage and ensure users are evaluated fairly,
        # sample a fraction of ratings per user
        train_indices = []
        test_indices = []

        grouped = self.ratings.groupby("user_id", group_keys=False)
        for _, group in grouped:
            n = len(group)
            if n < 5:
                # If very few ratings, put all in train
                train_indices.extend(group.index.tolist())
            else:
                n_test = max(1, int(n * self.test_size))
                shuffled_idx = rng.permutation(group.index.values)
                test_indices.extend(shuffled_idx[:n_test])
                train_indices.extend(shuffled_idx[n_test:])

        self.train_ratings = self.ratings.loc[train_indices].copy()
        self.test_ratings = self.ratings.loc[test_indices].copy()

        # Build fast lookup sets
        for row in self.train_ratings.itertuples():
            self.user_train_rated[row.user_id].add(row.movie_id)

        for row in self.test_ratings.itertuples():
            self.user_test_all[row.user_id].append(row.movie_id)
            if row.rating >= self.min_relevance_rating:
                self.user_test_relevant[row.user_id].append(row.movie_id)

        # Build contiguous index maps
        unique_users = sorted(self.ratings["user_id"].unique())
        unique_movies = sorted(self.movies["movie_id"].unique())

        self.user_to_idx = {u: i for i, u in enumerate(unique_users)}
        self.idx_to_user = {i: u for u, i in self.user_to_idx.items()}
        self.movie_to_idx = {m: i for i, m in enumerate(unique_movies)}
        self.idx_to_movie = {i: m for m, i in self.movie_to_idx.items()}

    def get_sparse_train_matrix(self) -> csr_matrix:
        """Constructs a sparse CSR user-item rating matrix for training."""
        rows = [self.user_to_idx[uid] for uid in self.train_ratings["user_id"]]
        cols = [self.movie_to_idx[mid] for mid in self.train_ratings["movie_id"]]
        vals = self.train_ratings["rating"].values.astype(np.float32)

        n_users = len(self.user_to_idx)
        n_movies = len(self.movie_to_idx)

        return csr_matrix((vals, (rows, cols)), shape=(n_users, n_movies))

    def get_user_history(self, user_id: int, top_n: int = 10) -> List[Dict]:
        """Returns the user's top-rated movies in the training set."""
        if user_id not in self.user_train_rated:
            return []
        user_rows = self.train_ratings[self.train_ratings["user_id"] == user_id]
        sorted_rows = user_rows.sort_values(by="rating", ascending=False).head(top_n)
        history = []
        for _, row in sorted_rows.iterrows():
            mid = int(row["movie_id"])
            history.append({
                "movie_id": mid,
                "title": self.movie_id_to_title.get(mid, "Unknown"),
                "rating": float(row["rating"]),
                "genres": self.movie_id_to_genres.get(mid, ""),
            })
        return history
