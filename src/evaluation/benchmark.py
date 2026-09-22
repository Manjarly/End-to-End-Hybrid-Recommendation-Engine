"""
Benchmark Suite for comparing Recommender Models side-by-side.
Calculates NDCG, MAP, MRR, Precision, Recall, Coverage, Diversity, and Novelty.
Generates publication-quality comparison charts.
"""

from pathlib import Path
from typing import Dict, List, Optional, Sequence
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .metrics import (
    precision_at_k,
    recall_at_k,
    ndcg_at_k,
    average_precision_at_k,
    mrr_at_k,
    hit_rate_at_k,
    catalog_coverage,
    novelty_at_k,
    intra_list_diversity_at_k,
)
from ..data.dataset import RecommendationDataset
from ..models.base import BaseRecommender


class BenchmarkSuite:
    """Executes multi-model comparative benchmarking and visualization."""

    def __init__(self, dataset: RecommendationDataset, k: int = 10, max_eval_users: int = 400):
        self.dataset = dataset
        self.k = k
        self.max_eval_users = max_eval_users
        self.results_df: Optional[pd.DataFrame] = None

    def evaluate_model(self, model: BaseRecommender) -> Dict[str, float]:
        """Evaluates a single recommender model on the test split."""
        eval_users = [
            u for u, items in self.dataset.user_test_relevant.items()
            if len(items) > 0 and u in self.dataset.user_train_rated
        ]

        if len(eval_users) > self.max_eval_users:
            rng = np.random.RandomState(42)
            eval_users = rng.choice(eval_users, size=self.max_eval_users, replace=False).tolist()

        precisions, recalls, ndcgs, maps, mrrs, hit_rates = [], [], [], [], [], []
        novelties, diversities = [], []
        all_recommended_items = set()

        total_catalog = len(self.dataset.movie_to_idx)
        total_users = len(self.dataset.user_to_idx)
        item_counts = self.dataset.train_ratings["movie_id"].value_counts().to_dict()

        for u in eval_users:
            relevant = set(self.dataset.user_test_relevant[u])
            recs_with_scores = model.recommend(u, n=self.k, exclude_rated=True)
            recs = [mid for mid, _ in recs_with_scores]

            all_recommended_items.update(recs)

            precisions.append(precision_at_k(recs, relevant, self.k))
            recalls.append(recall_at_k(recs, relevant, self.k))
            ndcgs.append(ndcg_at_k(recs, relevant, self.k))
            maps.append(average_precision_at_k(recs, relevant, self.k))
            mrrs.append(mrr_at_k(recs, relevant, self.k))
            hit_rates.append(hit_rate_at_k(recs, relevant, self.k))
            novelties.append(novelty_at_k(recs, item_counts, total_users, self.k))
            diversities.append(intra_list_diversity_at_k(recs, self.dataset.movie_id_to_genres, self.k))

        coverage = catalog_coverage(all_recommended_items, total_catalog)

        return {
            f"NDCG@{self.k}": float(np.mean(ndcgs)),
            f"Precision@{self.k}": float(np.mean(precisions)),
            f"Recall@{self.k}": float(np.mean(recalls)),
            f"MAP@{self.k}": float(np.mean(maps)),
            f"MRR@{self.k}": float(np.mean(mrrs)),
            f"HitRate@{self.k}": float(np.mean(hit_rates)),
            "Coverage": float(coverage),
            "Diversity": float(np.mean(diversities)),
            "Novelty": float(np.mean(novelties)),
        }

    def run_benchmark(self, models: Sequence[BaseRecommender]) -> pd.DataFrame:
        """Runs side-by-side benchmark across all supplied models."""
        rows = []
        for model in models:
            print(f"[BenchmarkSuite] Evaluating {model.name}...")
            metrics = self.evaluate_model(model)
            metrics["Model"] = model.name
            rows.append(metrics)

        df = pd.DataFrame(rows)
        # Reorder columns
        cols = ["Model", f"NDCG@{self.k}", f"Precision@{self.k}", f"Recall@{self.k}", f"MAP@{self.k}", f"MRR@{self.k}", "Coverage", "Diversity", "Novelty"]
        self.results_df = df[cols].sort_values(by=f"NDCG@{self.k}", ascending=False).reset_index(drop=True)
        return self.results_df

    def plot_benchmark(self, output_dir: Optional[Path] = None) -> Path:
        """Generates and saves visual comparison plots."""
        if self.results_df is None:
            raise RuntimeError("Run benchmark before plotting.")

        if output_dir is None:
            output_dir = Path(__file__).resolve().parent.parent.parent / "assets"
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "model_benchmark.png"

        sns.set_theme(style="whitegrid", font="sans-serif")
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Ranking Quality (NDCG & Precision)
        ax1 = axes[0, 0]
        sns.barplot(
            data=self.results_df,
            x="Model",
            y=f"NDCG@{self.k}",
            palette="mako",
            ax=ax1,
            hue="Model",
            legend=False,
        )
        ax1.set_title(f"Ranking Quality: NDCG@{self.k}", fontsize=13, fontweight="bold")
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=25, ha="right")
        ax1.set_ylim(0, max(self.results_df[f"NDCG@{self.k}"]) * 1.25)

        # 2. Precision & Recall
        ax2 = axes[0, 1]
        sns.barplot(
            data=self.results_df,
            x="Model",
            y=f"Precision@{self.k}",
            palette="crest",
            ax=ax2,
            hue="Model",
            legend=False,
        )
        ax2.set_title(f"Accuracy: Precision@{self.k}", fontsize=13, fontweight="bold")
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=25, ha="right")
        ax2.set_ylim(0, max(self.results_df[f"Precision@{self.k}"]) * 1.25)

        # 3. Catalog Coverage
        ax3 = axes[1, 0]
        sns.barplot(
            data=self.results_df,
            x="Model",
            y="Coverage",
            palette="rocket",
            ax=ax3,
            hue="Model",
            legend=False,
        )
        ax3.set_title("Catalog Exploration: Coverage", fontsize=13, fontweight="bold")
        ax3.set_xticklabels(ax3.get_xticklabels(), rotation=25, ha="right")
        ax3.set_ylim(0, max(self.results_df["Coverage"]) * 1.25)

        # 4. Diversity
        ax4 = axes[1, 1]
        sns.barplot(
            data=self.results_df,
            x="Model",
            y="Diversity",
            palette="magma",
            ax=ax4,
            hue="Model",
            legend=False,
        )
        ax4.set_title("Recommendation Diversity (Intra-List)", fontsize=13, fontweight="bold")
        ax4.set_xticklabels(ax4.get_xticklabels(), rotation=25, ha="right")
        ax4.set_ylim(0, 1.0)

        plt.suptitle("End-to-End Hybrid Recommendation Engine — Multi-Metric Benchmark", fontsize=16, fontweight="bold")
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"[BenchmarkSuite] Benchmark plot saved to: {out_path}")
        return out_path
