"""
Unit tests for data preprocessor and RecommendationDataset.
"""

import pandas as pd
import pytest
from src.data.preprocessor import DataPreprocessor
from src.data.dataset import RecommendationDataset


@pytest.fixture
def mock_raw_data():
    movies = pd.DataFrame({
        "movie_id": [1, 2, 3, 4, 5],
        "title": [
            "Toy Story (1995)",
            "Jumanji (1995)",
            "Heat (1995)",
            "Star Wars (1977)",
            "Matrix, The (1999)",
        ],
        "genres": [
            "Animation|Children's|Comedy",
            "Adventure|Children's|Fantasy",
            "Action|Crime|Thriller",
            "Action|Adventure|Sci-Fi",
            "Action|Sci-Fi|Thriller",
        ],
    })

    ratings = pd.DataFrame({
        "user_id": [1, 1, 1, 2, 2, 3, 3, 3, 3, 4],
        "movie_id": [1, 2, 4, 3, 4, 1, 4, 5, 2, 5],
        "rating": [5.0, 4.0, 5.0, 3.0, 4.0, 5.0, 5.0, 4.0, 3.0, 4.0],
        "timestamp": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
    })

    users = pd.DataFrame({
        "user_id": [1, 2, 3, 4],
        "gender": ["M", "F", "M", "F"],
        "age": [25, 35, 18, 50],
        "age_desc": ["25-34", "35-44", "18-24", "50-55"],
        "occupation": [12, 1, 4, 7],
        "occupation_desc": ["programmer", "academic/educator", "college/grad student", "executive/managerial"],
        "zip_code": ["90210", "10001", "85281", "94107"],
    })

    return movies, ratings, users


def test_preprocessor_enrich_movies(mock_raw_data):
    movies, ratings, _ = mock_raw_data
    preprocessor = DataPreprocessor()
    enriched = preprocessor._enrich_movies(movies, ratings)

    assert "year" in enriched.columns
    assert enriched.loc[enriched["movie_id"] == 1, "year"].values[0] == 1995
    assert enriched.loc[enriched["movie_id"] == 4, "decade"].values[0] == "1970s"
    assert "bayesian_rating" in enriched.columns
    assert "metadata_soup" in enriched.columns


def test_recommendation_dataset_splits(mock_raw_data):
    movies, ratings, users = mock_raw_data
    preprocessor = DataPreprocessor()
    enriched_movies = preprocessor._enrich_movies(movies, ratings)

    dataset = RecommendationDataset(enriched_movies, ratings, users, test_size=0.25, random_state=42)

    assert len(dataset.train_ratings) > 0
    assert len(dataset.movie_to_idx) == 5
    assert len(dataset.user_to_idx) == 4

    sparse_mat = dataset.get_sparse_train_matrix()
    assert sparse_mat.shape == (4, 5)

    history = dataset.get_user_history(1)
    assert len(history) > 0
