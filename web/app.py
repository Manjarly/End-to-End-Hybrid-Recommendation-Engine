"""
Flask backend application for the Interactive Hybrid Recommendation Dashboard.
Provides high-performance REST APIs for recommendations, movie search, personas, and metrics.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

from src.data.downloader import DatasetDownloader
from src.data.preprocessor import DataPreprocessor
from src.data.dataset import RecommendationDataset
from src.models.collaborative import BiasedSVDRecommender
from src.models.content_based import ContentBasedRecommender
from src.models.hybrid import HybridRecommender
from src.explanations.explainer import RecommendationExplainer


def sanitize_json(obj):
    """Recursively converts numpy scalars and arrays into native Python JSON types."""
    if isinstance(obj, dict):
        return {str(k): sanitize_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [sanitize_json(v) for v in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def create_app(dataset_name: str = "ml-latest-small") -> Flask:
    static_dir = Path(__file__).parent / "static"
    app = Flask(__name__, static_folder=str(static_dir), static_url_path="")

    # In-memory cached models and dataset
    print(f"[Web App] Initializing dataset: {dataset_name}...")
    downloader = DatasetDownloader()
    preprocessor = DataPreprocessor()

    if dataset_name == "ml-1m":
        downloader.ensure_rich_dataset()
        movies, ratings, users = preprocessor.load_ml_1m(enrich_with_tags=True)
    elif dataset_name == "ml-latest-small":
        downloader.download_and_extract("ml-latest-small")
        movies, ratings, users = preprocessor.load_ml_latest_small()
    else:
        downloader.download_and_extract("ml-100k")
        movies, ratings, users = preprocessor.load_ml_100k()

    dataset = RecommendationDataset(movies, ratings, users)
    print(f"[Web App] Training Hybrid Engine for dataset with {len(ratings):,} ratings...")
    cf = BiasedSVDRecommender(n_components=40)
    cb = ContentBasedRecommender()
    hybrid = HybridRecommender(cf_model=cf, cb_model=cb, alpha=0.65, fusion_strategy="rrf")
    hybrid.fit(dataset)
    explainer = RecommendationExplainer(dataset, hybrid)

    # Curated Viewer Profiles
    personas = [
        {
            "user_id": 50,
            "label": "Drama and Suspense Collection (Viewer 50)",
            "type": "mature",
            "bio": "Drawn to psychological mysteries, sharp drama, and character studies like American Beauty and Rosemary's Baby.",
            "age_desc": "25-34",
        },
        {
            "user_id": 1,
            "label": "Animation and Family Favorites (Viewer 1)",
            "type": "mature",
            "bio": "Enjoys classic animation, heartwarming adventures, and timeless cinema such as Toy Story and Pocahontas.",
            "age_desc": "Under 18",
        },
        {
            "user_id": 23,
            "label": "Action and Sci-Fi Selection (Viewer 23)",
            "type": "mature",
            "bio": "Prefers visionary science fiction, high-stakes adventures, and cinema like Star Wars and Terminator.",
            "age_desc": "35-44",
        },
        {
            "user_id": 2,
            "label": "Classic and Historical Masterpieces (Viewer 2)",
            "type": "mature",
            "bio": "Appreciates classic American cinema, golden-age masterworks, and compelling historical drama.",
            "age_desc": "56+",
        },
        {
            "user_id": 9999,
            "label": "New Viewer (Choose any movie you like)",
            "type": "cold_start",
            "bio": "Start from scratch. Pick any favorite film or explore top community classics.",
            "age_desc": "25-34",
        },
    ]

    @app.route("/")
    def index():
        return send_from_directory(str(static_dir), "index.html")

    @app.route("/api/stats", methods=["GET"])
    def get_stats():
        return jsonify({
            "dataset": dataset_name,
            "total_ratings": len(dataset.ratings),
            "total_users": len(dataset.users),
            "total_movies": len(dataset.movies),
            "train_ratings": len(dataset.train_ratings),
            "test_ratings": len(dataset.test_ratings),
            "density_pct": round((len(dataset.ratings) / (len(dataset.users) * len(dataset.movies))) * 100, 2),
        })

    @app.route("/api/personas", methods=["GET"])
    def get_personas():
        return jsonify(personas)

    @app.route("/api/user_history/<int:user_id>", methods=["GET"])
    def get_user_history(user_id: int):
        history = dataset.get_user_history(user_id, top_n=6)
        return jsonify(history)

    @app.route("/api/movies", methods=["GET"])
    def search_movies():
        q = request.args.get("q", "").strip().lower()
        if not q:
            # Return top 20 by popularity
            top_movies = dataset.movies.sort_values(by="popularity_score", ascending=False).head(20)
        else:
            matches = dataset.movies[dataset.movies["title"].str.lower().str.contains(q, regex=False)]
            top_movies = matches.head(25)

        results = []
        for row in top_movies.itertuples():
            mid = int(row.movie_id)
            results.append({
                "movie_id": mid,
                "title": row.title,
                "clean_title": getattr(row, "clean_title", row.title),
                "year": int(row.year) if pd.notnull(getattr(row, "year", None)) else None,
                "genres": row.genres.split("|"),
                "bayesian_rating": round(float(row.bayesian_rating), 2) if hasattr(row, "bayesian_rating") else 4.0,
                "rating_count": int(row.rating_count) if hasattr(row, "rating_count") else 0,
                "tags": getattr(row, "tags", ""),
                "tmdb_id": int(row.tmdbId) if hasattr(row, "tmdbId") and pd.notnull(row.tmdbId) else None,
            })
        return jsonify(results)

    @app.route("/api/recommend", methods=["POST"])
    def get_recommendations():
        data = request.get_json() or {}
        user_id = int(data.get("user_id", 50))
        seed_movie_id = data.get("seed_movie_id")
        if seed_movie_id is not None and str(seed_movie_id).strip():
            seed_movie_id = int(seed_movie_id)
        else:
            seed_movie_id = None

        alpha = float(data.get("alpha", 0.65))
        strategy = data.get("strategy", "rrf")
        apply_diversity = bool(data.get("apply_diversity", False))
        top_n = int(data.get("top_n", 10))
        age_desc = data.get("age_desc")

        recs = hybrid.recommend(
            user_id=user_id,
            n=top_n,
            seed_movie_id=seed_movie_id,
            alpha=alpha,
            strategy=strategy,
            age_desc=age_desc,
            apply_diversity=apply_diversity,
        )

        cards = []
        for rank, (mid, score) in enumerate(recs, 1):
            title = dataset.movie_id_to_title.get(mid, "Unknown")
            genres = dataset.movie_id_to_genres.get(mid, "").split("|")
            movie_rows = dataset.movies[dataset.movies["movie_id"] == mid]
            year = None
            b_rating = 4.0
            r_count = 0
            tmdb_id = None
            if not movie_rows.empty:
                m_row = movie_rows.iloc[0]
                year = int(m_row["year"]) if pd.notnull(m_row.get("year")) else None
                b_rating = round(float(m_row.get("bayesian_rating", 4.0)), 2)
                r_count = int(m_row.get("rating_count", 0))
                tmdb_id = int(m_row.get("tmdbId")) if pd.notnull(m_row.get("tmdbId")) else None

            explanation = explainer.explain(
                user_id, mid, seed_movie_id=seed_movie_id, score=score, age_desc=age_desc
            )

            cards.append({
                "rank": int(rank),
                "movie_id": int(mid),
                "title": str(title),
                "year": int(year) if year is not None else None,
                "genres": genres,
                "score": round(float(score), 4),
                "bayesian_rating": b_rating,
                "rating_count": r_count,
                "tmdb_id": tmdb_id,
                "badge": explanation["badge"],
                "badge_type": explanation["badge_type"],
                "reason": explanation["reason"],
                "highlight": explanation["highlight"],
            })

        return jsonify(sanitize_json({
            "user_id": user_id,
            "seed_movie_id": seed_movie_id,
            "alpha": alpha,
            "strategy": strategy,
            "apply_diversity": apply_diversity,
            "recommendations": cards,
        }))

    @app.route("/api/benchmark", methods=["GET"])
    def get_benchmark():
        # Precomputed high-fidelity metrics from benchmark suite
        metrics = [
            {"model": "Hybrid Engine (RRF)", "ndcg": 0.1328, "precision": 0.1095, "recall": 0.0540, "map": 0.0718, "mrr": 0.2849, "coverage": 0.0337, "diversity": 0.8588, "novelty": 2.91},
            {"model": "Biased SVD (k=40)", "ndcg": 0.1292, "precision": 0.1075, "recall": 0.0526, "map": 0.0697, "mrr": 0.2737, "coverage": 0.0301, "diversity": 0.8605, "novelty": 2.90},
            {"model": "Hybrid Engine (Weighted)", "ndcg": 0.1199, "precision": 0.0955, "recall": 0.0490, "map": 0.0621, "mrr": 0.2707, "coverage": 0.0464, "diversity": 0.8346, "novelty": 3.25},
            {"model": "Popularity (Bayesian)", "ndcg": 0.0874, "precision": 0.0775, "recall": 0.0430, "map": 0.0370, "mrr": 0.1970, "coverage": 0.0070, "diversity": 0.8882, "novelty": 2.61},
            {"model": "Content-Based (Soup)", "ndcg": 0.0299, "precision": 0.0255, "recall": 0.0210, "map": 0.0119, "mrr": 0.0642, "coverage": 0.0943, "diversity": 0.3831, "novelty": 6.36},
            {"model": "Random Recommender", "ndcg": 0.0045, "precision": 0.0050, "recall": 0.0016, "map": 0.0012, "mrr": 0.0115, "coverage": 0.4108, "diversity": 0.8265, "novelty": 6.79},
        ]
        return jsonify(metrics)

    return app
