"""
Unit tests for models: Baselines, CF, Content-Based, and Hybrid.
"""

import pandas as pd
import pytest
from src.data.preprocessor import DataPreprocessor
from src.data.dataset import RecommendationDataset
from src.models.baselines import RandomRecommender, PopularityRecommender, DemographicRecommender
from src.models.collaborative import BiasedSVDRecommender, ItemItemCFRecommender
from src.models.content_based import ContentBasedRecommender
from src.models.hybrid import HybridRecommender


@pytest.fixture
def mock_dataset():
    movies = pd.DataFrame({
        "movie_id": [1, 2, 3, 4, 5, 6],
        "title": [
            "Toy Story (1995)",
            "Jumanji (1995)",
            "Heat (1995)",
            "Star Wars (1977)",
            "Matrix, The (1999)",
            "Alien (1979)",
        ],
        "genres": [
            "Animation|Children's|Comedy",
            "Adventure|Children's|Fantasy",
            "Action|Crime|Thriller",
            "Action|Adventure|Sci-Fi",
            "Action|Sci-Fi|Thriller",
            "Action|Horror|Sci-Fi",
        ],
    })

    ratings = pd.DataFrame({
        "user_id": [1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5],
        "movie_id": [1, 2, 4, 5, 3, 4, 6, 1, 4, 5, 2, 5, 6, 1, 3],
        "rating": [5.0, 4.0, 5.0, 4.5, 3.0, 4.0, 5.0, 5.0, 5.0, 4.0, 3.0, 4.0, 4.5, 5.0, 4.0],
        "timestamp": list(range(100, 115)),
    })

    users = pd.DataFrame({
        "user_id": [1, 2, 3, 4, 5],
        "gender": ["M", "F", "M", "F", "M"],
        "age": [25, 35, 18, 50, 25],
        "age_desc": ["25-34", "35-44", "18-24", "50-55", "25-34"],
        "occupation": [12, 1, 4, 7, 12],
        "occupation_desc": ["programmer", "academic", "student", "executive", "programmer"],
        "zip_code": ["90210", "10001", "85281", "94107", "90210"],
    })

    preprocessor = DataPreprocessor()
    enriched_movies = preprocessor._enrich_movies(movies, ratings)
    return RecommendationDataset(enriched_movies, ratings, users, test_size=0.2, random_state=42)


def test_baselines(mock_dataset):
    rand_m = RandomRecommender().fit(mock_dataset)
    recs = rand_m.recommend(1, n=3)
    assert len(recs) <= 3

    pop_m = PopularityRecommender().fit(mock_dataset)
    recs_pop = pop_m.recommend(1, n=3)
    assert len(recs_pop) <= 3
    # Top popular item should have high score
    assert recs_pop[0][1] > 0

    demo_m = DemographicRecommender().fit(mock_dataset)
    recs_demo = demo_m.recommend(1, n=3)
    assert len(recs_demo) <= 3
    recs_age = demo_m.recommend(1, n=3, age_desc="25-34")
    assert len(recs_age) <= 3


def test_biased_svd(mock_dataset):
    svd_m = BiasedSVDRecommender(n_components=2).fit(mock_dataset)
    recs = svd_m.recommend(1, n=3, exclude_rated=True)
    assert len(recs) <= 3
    score = svd_m.predict_score(1, 3)
    assert 1.0 <= score <= 5.0


def test_content_based(mock_dataset):
    cb_m = ContentBasedRecommender().fit(mock_dataset)
    # Similar to Star Wars (id=4, Sci-Fi) should be Matrix or Alien
    sims = cb_m.get_similar_movies(4, top_n=2)
    assert len(sims) > 0
    top_similar_id = sims[0][0]
    assert top_similar_id in [5, 6]  # Matrix or Alien


def test_hybrid_recommender(mock_dataset):
    cf = BiasedSVDRecommender(n_components=2)
    cb = ContentBasedRecommender()
    hybrid = HybridRecommender(cf_model=cf, cb_model=cb, fusion_strategy="rrf").fit(mock_dataset)

    # 1. Existing user
    recs_rrf = hybrid.recommend(1, n=3, exclude_rated=True)
    assert len(recs_rrf) <= 3

    # 2. Weighted strategy
    recs_weighted = hybrid.recommend(1, n=3, strategy="weighted", alpha=0.7)
    assert len(recs_weighted) <= 3

    # 3. Cold Start user (user_id=999) with seed movie
    recs_cold = hybrid.recommend(999, n=3, seed_movie_id=4)
    assert len(recs_cold) <= 3
    assert any(m in [5, 6] for m, _ in recs_cold)

    # 4. Demographic hybrid recommendation (Age group)
    recs_demo = hybrid.recommend(999, n=3, age_desc="25-34")
    assert len(recs_demo) <= 3
