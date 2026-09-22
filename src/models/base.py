"""
Abstract Base Class for Recommender Models.
Defines standard interface for fitting, score prediction, and top-N recommendation.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
from ..data.dataset import RecommendationDataset


class BaseRecommender(ABC):
    """Abstract interface for all recommendation algorithms."""

    def __init__(self, name: str):
        self.name = name
        self.is_fitted = False

    @abstractmethod
    def fit(self, dataset: RecommendationDataset) -> "BaseRecommender":
        """Fit the recommender on the training dataset."""
        pass

    @abstractmethod
    def recommend(
        self,
        user_id: int,
        n: int = 10,
        exclude_rated: bool = True,
        seed_movie_id: Optional[int] = None,
    ) -> List[Tuple[int, float]]:
        """
        Recommends top-n movie IDs along with their predicted score or ranking metric.
        Returns:
            List of (movie_id, score) tuples sorted by score descending.
        """
        pass

    @abstractmethod
    def predict_score(self, user_id: int, movie_id: int) -> float:
        """Predicts the rating or affinity score of user_id for movie_id."""
        pass
