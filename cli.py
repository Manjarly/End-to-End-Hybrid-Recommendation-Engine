"""
Unified Command-Line Interface (CLI) for the Hybrid Recommendation Engine.
Supports data downloading, model training, evaluation, benchmarking, live demo, and web serving.
"""

import argparse
import sys
from pathlib import Path
import pandas as pd

from src.data.downloader import DatasetDownloader
from src.data.preprocessor import DataPreprocessor
from src.data.dataset import RecommendationDataset
from src.models.baselines import RandomRecommender, PopularityRecommender, DemographicRecommender
from src.models.collaborative import BiasedSVDRecommender, ItemItemCFRecommender
from src.models.content_based import ContentBasedRecommender
from src.models.hybrid import HybridRecommender
from src.evaluation.benchmark import BenchmarkSuite
from src.explanations.explainer import RecommendationExplainer


def cmd_download(args):
    """Downloads dataset and required enrichment files."""
    print(f"=== DOWNLOADING DATASET: {args.dataset} ===")
    downloader = DatasetDownloader()
    if args.dataset == "ml-1m":
        downloader.ensure_rich_dataset()
    else:
        downloader.download_and_extract(args.dataset)
    print("Download and setup finished successfully!")


def load_dataset(dataset_name: str = "ml-1m") -> RecommendationDataset:
    """Helper to download if needed and build RecommendationDataset."""
    downloader = DatasetDownloader()
    preprocessor = DataPreprocessor()

    if dataset_name == "ml-1m":
        downloader.ensure_rich_dataset()
        movies, ratings, users = preprocessor.load_ml_1m(enrich_with_tags=True)
    elif dataset_name == "ml-latest-small":
        downloader.download_and_extract("ml-latest-small")
        movies, ratings, users = preprocessor.load_ml_latest_small()
    elif dataset_name == "ml-100k":
        downloader.download_and_extract("ml-100k")
        movies, ratings, users = preprocessor.load_ml_100k()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    print(f"[Data] Loaded {len(ratings):,} ratings, {len(users):,} users, {len(movies):,} movies.")
    dataset = RecommendationDataset(movies, ratings, users)
    return dataset


def cmd_benchmark(args):
    """Runs rigorous multi-model benchmarking."""
    print(f"=== RUNNING BENCHMARK (Dataset: {args.dataset}, K={args.k}) ===")
    dataset = load_dataset(args.dataset)

    print("\nFitting models for benchmarking...")
    random_m = RandomRecommender().fit(dataset)
    pop_m = PopularityRecommender(use_bayesian=True).fit(dataset)
    cb_m = ContentBasedRecommender().fit(dataset)
    cf_m = BiasedSVDRecommender(n_components=args.n_components).fit(dataset)
    hybrid_rrf = HybridRecommender(cf_model=cf_m, cb_model=cb_m, fusion_strategy="rrf").fit(dataset)
    hybrid_weighted = HybridRecommender(cf_model=cf_m, cb_model=cb_m, fusion_strategy="weighted", alpha=0.65).fit(dataset)

    models = [random_m, pop_m, cb_m, cf_m, hybrid_weighted, hybrid_rrf]

    suite = BenchmarkSuite(dataset, k=args.k, max_eval_users=args.max_users)
    results = suite.run_benchmark(models)

    print("\n" + "=" * 80)
    print("FINAL BENCHMARK RESULTS")
    print("=" * 80)
    try:
        print(results.to_markdown(index=False))
    except Exception:
        print(results.to_string(index=False))
    print("=" * 80 + "\n")

    if args.plot:
        plot_path = suite.plot_benchmark()
        print(f"Benchmark plot generated: {plot_path}")


def cmd_demo(args):
    """Runs interactive recommendation demonstration."""
    print(f"=== HYBRID RECOMMENDATION DEMO (Dataset: {args.dataset}) ===")
    dataset = load_dataset(args.dataset)

    hybrid = HybridRecommender(alpha=args.alpha, fusion_strategy=args.strategy).fit(dataset)
    explainer = RecommendationExplainer(dataset, hybrid)

    seed_id = None
    if args.movie:
        # Search movie
        matches = dataset.movies[dataset.movies["title"].str.contains(args.movie, case=False, regex=False)]
        if not matches.empty:
            seed_id = int(matches.iloc[0]["movie_id"])
            print(f"Matched seed movie: '{matches.iloc[0]['title']}' (ID: {seed_id})")
        else:
            print(f"Warning: Could not find movie matching '{args.movie}'. Proceeding without seed.")

    user_id = args.user_id
    user_history = dataset.get_user_history(user_id, top_n=5)

    print(f"\nUser ID: {user_id}")
    if user_history:
        print("User's Top-Rated Movies:")
        for idx, h in enumerate(user_history, 1):
            print(f"  {idx}. {h['title']} (★{h['rating']:.1f}) - {h['genres']}")
    else:
        print("  -> New User Profile (Cold Start)")

    recs = hybrid.recommend(
        user_id=user_id,
        n=args.n,
        seed_movie_id=seed_id,
        alpha=args.alpha,
        strategy=args.strategy,
        apply_diversity=args.diversity,
    )

    print(f"\nTop-{args.n} Recommendations ({args.strategy.upper()} Fusion, alpha={args.alpha}):")
    print("-" * 75)
    for rank, (mid, score) in enumerate(recs, 1):
        title = dataset.movie_id_to_title.get(mid, "Unknown")
        genres = dataset.movie_id_to_genres.get(mid, "")
        exp = explainer.explain(user_id, mid, seed_movie_id=seed_id, score=score)
        print(f"{rank:2d}. {title} [{genres}]")
        print(f"    Rationale: [{exp['badge']}] {exp['reason']}")
    print("-" * 75)


def cmd_serve(args):
    """Starts the interactive web application server."""
    from web.app import create_app
    print(f"Starting Recommendation Dashboard at http://{args.host}:{args.port}")
    app = create_app(dataset_name=args.dataset)
    app.run(host=args.host, port=args.port, debug=False)


def main():
    parser = argparse.ArgumentParser(description="End-to-End Hybrid Recommendation Engine CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # download
    p_dl = subparsers.add_parser("download", help="Download datasets")
    p_dl.add_argument("--dataset", choices=["ml-1m", "ml-latest-small", "ml-100k"], default="ml-1m")

    # benchmark
    p_bench = subparsers.add_parser("benchmark", help="Run multi-model benchmark")
    p_bench.add_argument("--dataset", choices=["ml-1m", "ml-latest-small", "ml-100k"], default="ml-1m")
    p_bench.add_argument("-k", type=int, default=10, help="Top-K evaluation cutoff")
    p_bench.add_argument("--n-components", type=int, default=40, help="SVD latent dimensions")
    p_bench.add_argument("--max-users", type=int, default=300, help="Max test users for evaluation")
    p_bench.add_argument("--plot", action="store_true", default=True, help="Save evaluation plot")

    # demo
    p_demo = subparsers.add_parser("demo", help="Generate recommendations for a user")
    p_demo.add_argument("--dataset", choices=["ml-1m", "ml-latest-small", "ml-100k"], default="ml-1m")
    p_demo.add_argument("--user-id", type=int, default=50, help="User ID (or 9999 for cold-start)")
    p_demo.add_argument("--movie", type=str, default="Star Wars (1977)", help="Seed movie name or keyword")
    p_demo.add_argument("-n", type=int, default=10, help="Number of recommendations")
    p_demo.add_argument("--alpha", type=float, default=0.65, help="CF weight (0.0 to 1.0)")
    p_demo.add_argument("--strategy", choices=["rrf", "weighted"], default="rrf", help="Fusion strategy")
    p_demo.add_argument("--diversity", action="store_true", help="Apply MMR diversity re-ranking")

    # serve
    p_serve = subparsers.add_parser("serve", help="Launch interactive web app")
    p_serve.add_argument("--dataset", choices=["ml-1m", "ml-latest-small", "ml-100k"], default="ml-latest-small")
    p_serve.add_argument("--host", type=str, default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=5050)

    args = parser.parse_args()
    if args.command == "download":
        cmd_download(args)
    elif args.command == "benchmark":
        cmd_benchmark(args)
    elif args.command == "demo":
        cmd_demo(args)
    elif args.command == "serve":
        cmd_serve(args)


if __name__ == "__main__":
    main()
