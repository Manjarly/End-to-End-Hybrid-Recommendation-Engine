"""
Generates the next-generation End_to_End_Hybrid_Recommendation_Engine.ipynb notebook.
"""

import json
from pathlib import Path

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 🎬 Next-Generation End-to-End Hybrid Recommendation Engine\n",
                "\n",
                "### Production-Scale Recommender System on **MovieLens 1M** (1,000,209 Ratings · 6,040 Users · 3,883 Movies)\n",
                "\n",
                "This project implements a state-of-the-art Hybrid Recommendation Engine combining:\n",
                "1. **Biased Matrix Factorization (Collaborative Filtering)**: Regularized SVD with global, user, and item bias centering to avoid zero-fill penalties.\n",
                "2. **Metadata-Soup Content-Based Filtering**: TF-IDF vectorization across titles, genres, release decades, and aggregated user tags.\n",
                "3. **Reciprocal Rank Fusion (RRF) & Weighted Score Blending**: Industry gold-standard rank aggregation immune to score calibration discrepancies.\n",
                "4. **Dynamic Cold-Start & Demographic Routing**: Intelligent fallback to content similarity and demographic priors for brand-new users.\n",
                "5. **Comprehensive Ranking Evaluation**: Benchmarked across **NDCG@10, Precision@10, Recall@10, MAP@10, MRR@10, Catalog Coverage, Diversity, and Novelty**."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# ===================================================================\n",
                "# 1. Environment Setup & Core Dependencies\n",
                "# ===================================================================\n",
                "import os\n",
                "import sys\n",
                "from pathlib import Path\n",
                "\n",
                "import numpy as np\n",
                "import pandas as pd\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "\n",
                "# Add project root to sys.path\n",
                "project_root = Path.cwd()\n",
                "if str(project_root) not in sys.path:\n",
                "    sys.path.insert(0, str(project_root))\n",
                "\n",
                "from src.data.downloader import DatasetDownloader\n",
                "from src.data.preprocessor import DataPreprocessor\n",
                "from src.data.dataset import RecommendationDataset\n",
                "from src.models.baselines import RandomRecommender, PopularityRecommender, DemographicRecommender\n",
                "from src.models.collaborative import BiasedSVDRecommender\n",
                "from src.models.content_based import ContentBasedRecommender\n",
                "from src.models.hybrid import HybridRecommender\n",
                "from src.evaluation.benchmark import BenchmarkSuite\n",
                "from src.explanations.explainer import RecommendationExplainer\n",
                "\n",
                "sns.set_theme(style='whitegrid')\n",
                "print(\"Environment initialized successfully!\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Ingesting & Exploring MovieLens 1M + Rich Tags\n",
                "\n",
                "We download and cache **MovieLens 1M** (10x larger than the legacy 100K dataset) and enrich it with user-submitted tags and TMDB links from **MovieLens Latest**."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Download and load MovieLens 1M\n",
                "downloader = DatasetDownloader()\n",
                "downloader.ensure_rich_dataset()\n",
                "\n",
                "preprocessor = DataPreprocessor()\n",
                "movies_df, ratings_df, users_df = preprocessor.load_ml_1m(enrich_with_tags=True)\n",
                "\n",
                "print(f\"Loaded {len(ratings_df):,} ratings from {len(users_df):,} users across {len(movies_df):,} movies.\")\n",
                "movies_df[['movie_id', 'title', 'genres', 'year', 'decade', 'bayesian_rating', 'rating_count']].head(5)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Visualize Dataset Distributions\n",
                "fig, axes = plt.subplots(1, 3, figsize=(18, 5))\n",
                "\n",
                "# Rating Distribution\n",
                "sns.countplot(x='rating', data=ratings_df, palette='viridis', ax=axes[0], hue='rating', legend=False)\n",
                "axes[0].set_title(\"Distribution of Star Ratings\", fontsize=14, fontweight='bold')\n",
                "axes[0].set_xlabel(\"Rating (Stars)\")\n",
                "axes[0].set_ylabel(\"Count\")\n",
                "\n",
                "# Top Movie Genres\n",
                "all_genres = [g for sublist in movies_df['genres'].str.split('|') for g in sublist]\n",
                "genre_counts = pd.Series(all_genres).value_counts().head(10)\n",
                "sns.barplot(x=genre_counts.values, y=genre_counts.index, palette='mako', ax=axes[1], hue=genre_counts.index, legend=False)\n",
                "axes[1].set_title(\"Top 10 Movie Genres\", fontsize=14, fontweight='bold')\n",
                "axes[1].set_xlabel(\"Number of Movies\")\n",
                "\n",
                "# User Ratings Count Distribution (Log Scale)\n",
                "user_counts = ratings_df['user_id'].value_counts()\n",
                "axes[2].hist(user_counts, bins=40, color='#8b5cf6', edgecolor='white')\n",
                "axes[2].set_title(\"User Activity Distribution\", fontsize=14, fontweight='bold')\n",
                "axes[2].set_xlabel(\"Ratings per User\")\n",
                "axes[2].set_ylabel(\"Number of Users\")\n",
                "axes[2].set_yscale('log')\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Train / Test Stratification & Partitioning\n",
                "\n",
                "To prevent data leakage and evaluate fairly across all users, we apply **user-stratified temporal/random splitting** (80% Train / 20% Test)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "dataset = RecommendationDataset(movies_df, ratings_df, users_df, test_size=0.20, random_state=42)\n",
                "print(f\"Training set: {len(dataset.train_ratings):,} ratings\")\n",
                "print(f\"Test set:     {len(dataset.test_ratings):,} ratings\")\n",
                "print(f\"Test users with relevant items (>=4.0★): {len(dataset.user_test_relevant):,}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Collaborative Filtering: Biased Matrix Factorization\n",
                "\n",
                "Rather than naive `pivot().fillna(0)` which distorts recommendations by penalizing unobserved movies as 0-star ratings, we implement **Biased Matrix Factorization**:\n",
                "$$\\hat{r}_{u,i} = \\mu + b_u + b_i + P_u \\cdot Q_i^T$$\n",
                "where $\\mu$ is the global average, $b_u$ and $b_i$ are regularized user and movie biases, and $P, Q$ are latent factor vectors learned via regularized SVD on observed residuals."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "cf_model = BiasedSVDRecommender(n_components=40, random_state=42)\n",
                "cf_model.fit(dataset)\n",
                "print(f\"Global rating mean: {cf_model.global_mean:.2f}★\")\n",
                "print(f\"User factors shape: {cf_model.user_factors.shape}\")\n",
                "print(f\"Item factors shape: {cf_model.item_factors.shape}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Content-Based Filtering: Metadata Soup & TF-IDF\n",
                "\n",
                "We construct a rich multi-field metadata soup combining clean movie title, genres, release decade, and user tags, followed by sublinear TF-IDF vectorization and cosine similarity indexing."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "cb_model = ContentBasedRecommender(max_features=10000, ngram_range=(1, 2))\n",
                "cb_model.fit(dataset)\n",
                "print(f\"TF-IDF Matrix Vocabulary Size: {cb_model.tfidf_matrix.shape[1]:,} features\")\n",
                "\n",
                "# Inspect similar movies to 'Matrix, The (1999)'\n",
                "matrix_id = dataset.title_to_movie_id.get('Matrix, The (1999)', 2571)\n",
                "similar_to_matrix = cb_model.get_similar_movies(matrix_id, top_n=5)\n",
                "print(f\"\\nTop 5 Content-Similar Movies to 'Matrix, The (1999)':\")\n",
                "for mid, sim in similar_to_matrix:\n",
                "    print(f\"  • {dataset.movie_id_to_title.get(mid)} (Cosine Sim: {sim:.3f})\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 6. Hybrid Recommender Engine: Reciprocal Rank Fusion & Dynamic Cold-Start\n",
                "\n",
                "The Hybrid Engine integrates:\n",
                "1. **Reciprocal Rank Fusion (RRF)**:\n",
                "   $$RRF(i) = \\frac{w_{cf}}{k + \\text{rank}_{cf}(i)} + \\frac{w_{cb}}{k + \\text{rank}_{cb}(i)}$$\n",
                "2. **Dynamic Cold-Start Handling**: If the user has 0 ratings, smoothly falls back to seed-movie content similarity or demographic priors."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "hybrid_engine = HybridRecommender(\n",
                "    cf_model=cf_model,\n",
                "    cb_model=cb_model,\n",
                "    alpha=0.65,\n",
                "    fusion_strategy='rrf',\n",
                ")\n",
                "hybrid_engine.fit(dataset)\n",
                "explainer = RecommendationExplainer(dataset, hybrid_engine)\n",
                "print(\"Hybrid Engine & Explainer ready!\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 7. Interactive Demonstration: Existing User vs. Cold-Start User"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Case A: Mature User (User ID = 50)\n",
                "print(\"=\" * 75)\n",
                "print(\"CASE A: RECOMMENDATIONS FOR MATURE USER (#50)\")\n",
                "print(\"=\" * 75)\n",
                "history = dataset.get_user_history(50, top_n=4)\n",
                "print(\"User's Top Historical Favorites:\")\n",
                "for h in history:\n",
                "    print(f\"  • {h['title']} (★{h['rating']:.1f}) [{h['genres']}]\")\n",
                "\n",
                "recs_user50 = hybrid_engine.recommend(50, n=7)\n",
                "print(\"\\nTop Hybrid Recommendations:\")\n",
                "for rank, (mid, score) in enumerate(recs_user50, 1):\n",
                "    title = dataset.movie_id_to_title.get(mid)\n",
                "    genres = dataset.movie_id_to_genres.get(mid)\n",
                "    exp = explainer.explain(50, mid, score=score)\n",
                "    print(f\"  {rank}. {title} [{genres}]\")\n",
                "    print(f\"     [{exp['badge']}] {exp['reason']}\")\n",
                "\n",
                "# Case B: Cold-Start User with Seed Movie\n",
                "print(\"\\n\" + \"=\" * 75)\n",
                "print(\"CASE B: NEW USER (COLD START) ANCHORED ON 'Toy Story (1995)'\")\n",
                "print(\"=\" * 75)\n",
                "recs_cold = hybrid_engine.recommend(9999, n=5, seed_movie_id=1)\n",
                "for rank, (mid, score) in enumerate(recs_cold, 1):\n",
                "    title = dataset.movie_id_to_title.get(mid)\n",
                "    genres = dataset.movie_id_to_genres.get(mid)\n",
                "    exp = explainer.explain(9999, mid, seed_movie_id=1, score=score)\n",
                "    print(f\"  {rank}. {title} [{genres}]\")\n",
                "    print(f\"     [{exp['badge']}] {exp['reason']}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 8. Multi-Model Benchmarking & Evaluation\n",
                "\n",
                "We benchmark 6 distinct algorithms on the test set across ranking accuracy (**NDCG@10, Precision@10, Recall@10, MAP@10, MRR@10**) and beyond-accuracy metrics (**Catalog Coverage, Diversity, Novelty**)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Run Benchmark Suite\n",
                "suite = BenchmarkSuite(dataset, k=10, max_eval_users=200)\n",
                "\n",
                "random_m = RandomRecommender().fit(dataset)\n",
                "pop_m = PopularityRecommender(use_bayesian=True).fit(dataset)\n",
                "hybrid_weighted = HybridRecommender(cf_model=cf_model, cb_model=cb_model, fusion_strategy='weighted', alpha=0.65).fit(dataset)\n",
                "\n",
                "models_to_test = [random_m, pop_m, cb_model, cf_model, hybrid_weighted, hybrid_engine]\n",
                "benchmark_results = suite.run_benchmark(models_to_test)\n",
                "\n",
                "display(benchmark_results)\n",
                "\n",
                "# Generate comparison plots\n",
                "suite.plot_benchmark(output_dir=project_root / 'assets')\n",
                "from IPython.display import Image\n",
                "Image(filename=str(project_root / 'assets' / 'model_benchmark.png'))"
            ]
        }
    ],
    "metadata": {
        "language_info": {
            "name": "python",
            "version": "3.12.5"
        },
        "orig_nbformat": 4
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

target_path = Path("End_to_End_Hybrid_Recommendation_Engine.ipynb")
with open(target_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print(f"Upgraded notebook written to {target_path} successfully!")
