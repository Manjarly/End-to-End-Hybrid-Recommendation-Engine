"""
Data preprocessor and feature engineering pipeline for MovieLens datasets.
Enriches raw ratings, movie metadata, tags, and demographic data.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np


OCCUPATION_MAP: Dict[int, str] = {
    0: "other", 1: "academic/educator", 2: "artist", 3: "clerical/admin",
    4: "college/grad student", 5: "customer service", 6: "doctor/health care",
    7: "executive/managerial", 8: "farmer", 9: "homemaker", 10: "K-12 student",
    11: "lawyer", 12: "programmer", 13: "retired", 14: "sales/marketing",
    15: "scientist", 16: "self-employed", 17: "technician/engineer",
    18: "tradesman/craftsman", 19: "unemployed", 20: "writer",
}

AGE_MAP: Dict[int, str] = {
    1: "Under 18", 18: "18-24", 25: "25-34", 35: "35-44",
    45: "45-49", 50: "50-55", 56: "56+",
}


class DataPreprocessor:
    """Preprocesses and enriches MovieLens data for hybrid recommendations."""

    def __init__(self, data_dir: Optional[Path] = None):
        if data_dir is None:
            self.data_dir = Path(__file__).resolve().parent.parent.parent / "data"
        else:
            self.data_dir = Path(data_dir)

    def load_ml_1m(
        self,
        enrich_with_tags: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Loads MovieLens 1M dataset: movies, ratings, users.
        Returns:
            movies_df: pd.DataFrame
            ratings_df: pd.DataFrame
            users_df: pd.DataFrame
        """
        ml1m_dir = self.data_dir / "ml-1m"
        if not ml1m_dir.exists():
            raise FileNotFoundError(f"MovieLens 1M directory not found at {ml1m_dir}")

        # 1. Load movies
        movies_file = ml1m_dir / "movies.dat"
        movies = pd.read_csv(
            movies_file,
            sep="::",
            engine="python",
            names=["movie_id", "title", "genres"],
            encoding="latin-1",
            dtype={"movie_id": int, "title": str, "genres": str},
        )

        # 2. Load ratings
        ratings_file = ml1m_dir / "ratings.dat"
        ratings = pd.read_csv(
            ratings_file,
            sep="::",
            engine="python",
            names=["user_id", "movie_id", "rating", "timestamp"],
            encoding="latin-1",
            dtype={"user_id": int, "movie_id": int, "rating": float, "timestamp": int},
        )

        # 3. Load users
        users_file = ml1m_dir / "users.dat"
        users = pd.read_csv(
            users_file,
            sep="::",
            engine="python",
            names=["user_id", "gender", "age", "occupation", "zip_code"],
            encoding="latin-1",
            dtype={"user_id": int, "gender": str, "age": int, "occupation": int, "zip_code": str},
        )

        users["age_desc"] = users["age"].map(AGE_MAP).fillna("Unknown")
        users["occupation_desc"] = users["occupation"].map(OCCUPATION_MAP).fillna("Other")

        # 4. Feature Engineering on Movies
        movies = self._enrich_movies(movies, ratings)

        # 5. Enrich with tags if available
        if enrich_with_tags:
            movies = self._enrich_with_external_tags(movies)

        return movies, ratings, users

    def load_ml_100k(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Loads legacy MovieLens 100K dataset for comparison and backward compatibility."""
        ml100k_dir = self.data_dir / "ml-100k"
        if not ml100k_dir.exists():
            # Check repo's Data-Set directory
            repo_dataset = Path(__file__).resolve().parent.parent.parent / "Data-Set"
            if repo_dataset.exists() and (repo_dataset / "u.data").exists():
                ml100k_dir = repo_dataset
            else:
                raise FileNotFoundError(f"MovieLens 100k directory not found at {ml100k_dir}")

        genre_cols = [
            "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
            "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
            "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
        ]
        movie_cols = ["movie_id", "title", "release_date", "video_release_date", "IMDb_URL"] + genre_cols

        movies_raw = pd.read_csv(
            ml100k_dir / "u.item",
            sep="|",
            names=movie_cols,
            encoding="latin-1",
        )

        def make_genres_str(row):
            active = [g for g in genre_cols if row[g] == 1]
            return "|".join(active) if active else "unknown"

        movies = pd.DataFrame({
            "movie_id": movies_raw["movie_id"],
            "title": movies_raw["title"],
            "genres": movies_raw.apply(make_genres_str, axis=1),
        })

        ratings = pd.read_csv(
            ml100k_dir / "u.data",
            sep="\t",
            names=["user_id", "movie_id", "rating", "timestamp"],
        )

        users = pd.read_csv(
            ml100k_dir / "u.user",
            sep="|",
            names=["user_id", "age", "gender", "occupation", "zip_code"],
        )

        movies = self._enrich_movies(movies, ratings)
        return movies, ratings, users

    def load_ml_latest_small(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Loads modern MovieLens Latest Small dataset (goes up to 2018; includes Iron Man, Avengers, etc.)."""
        ml_dir = self.data_dir / "ml-latest-small"
        if not ml_dir.exists():
            raise FileNotFoundError(f"MovieLens Latest Small directory not found at {ml_dir}")

        movies = pd.read_csv(ml_dir / "movies.csv").rename(columns={"movieId": "movie_id"})
        ratings = pd.read_csv(ml_dir / "ratings.csv").rename(columns={"movieId": "movie_id", "userId": "user_id"})

        unique_uids = ratings["user_id"].unique()
        users_df = pd.DataFrame({"user_id": unique_uids})

        ml1m_users_file = self.data_dir / "ml-1m" / "users.dat"
        if ml1m_users_file.exists():
            u_1m = pd.read_csv(
                ml1m_users_file,
                sep="::",
                engine="python",
                names=["user_id", "gender", "age", "occupation", "zip_code"],
                encoding="latin-1",
                dtype={"user_id": int, "gender": str, "age": int, "occupation": int, "zip_code": str},
            )
            u_1m["age_desc"] = u_1m["age"].map(AGE_MAP).fillna("Unknown")
            u_1m["occupation_desc"] = u_1m["occupation"].map(OCCUPATION_MAP).fillna("Other")
            users = users_df.merge(u_1m, on="user_id", how="left")
            users["gender"] = users["gender"].fillna("M")
            users["age"] = users["age"].fillna(25).astype(int)
            users["age_desc"] = users["age_desc"].fillna("25-34")
            users["occupation"] = users["occupation"].fillna(0).astype(int)
            users["occupation_desc"] = users["occupation_desc"].fillna("Other")
            users["zip_code"] = users["zip_code"].fillna("00000")
        else:
            users = pd.DataFrame({
                "user_id": unique_uids,
                "gender": "M",
                "age": 25,
                "age_desc": "25-34",
                "occupation": 0,
                "occupation_desc": "Other",
                "zip_code": "00000",
            })

        movies = self._enrich_movies(movies, ratings)
        movies = self._enrich_with_external_tags(movies)
        return movies, ratings, users

    def _enrich_movies(self, movies: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
        """Extracts release year, computes Bayesian rating, vote count, and popularity score."""
        movies = movies.copy()

        # Extract year from title: e.g. "Toy Story (1995)" -> 1995
        year_pattern = r"\((\d{4})\)"
        movies["year"] = movies["title"].str.extract(year_pattern)[0].astype(float)
        # Compute decade: e.g. 1995 -> "1990s"
        movies["decade"] = movies["year"].apply(
            lambda y: f"{int((y // 10) * 10)}s" if pd.notnull(y) else "unknown"
        )
        # Clean title without year
        movies["clean_title"] = movies["title"].str.replace(year_pattern, "", regex=True).str.strip()

        # Compute rating statistics per movie
        stats = ratings.groupby("movie_id").agg(
            rating_count=("rating", "count"),
            rating_mean=("rating", "mean"),
        ).reset_index()

        movies = movies.merge(stats, on="movie_id", how="left")
        movies["rating_count"] = movies["rating_count"].fillna(0).astype(int)
        movies["rating_mean"] = movies["rating_mean"].fillna(0.0)

        # Bayesian Average Rating: WR = (v / (v + m)) * R + (m / (v + m)) * C
        # where v is vote count, R is mean rating, C is global mean rating, m is threshold
        global_mean = ratings["rating"].mean()
        m_threshold = 25  # minimum votes required for high confidence
        v = movies["rating_count"]
        R = movies["rating_mean"]
        movies["bayesian_rating"] = (v / (v + m_threshold)) * R + (m_threshold / (v + m_threshold)) * global_mean

        # Popularity score: log-scaled rating count weighted by bayesian score
        movies["popularity_score"] = np.log1p(movies["rating_count"]) * movies["bayesian_rating"]

        # Default tag string and metadata soup
        # Weight genres 3x so semantic genre and thematic kinship dominate title words
        movies["tags"] = ""
        genre_tokens = movies["genres"].str.replace("|", " ", regex=False)
        movies["metadata_soup"] = (
            (genre_tokens + " ") * 3
            + movies["clean_title"]
            + " "
            + movies["decade"]
        )

        return movies

    def _enrich_with_external_tags(self, movies: pd.DataFrame) -> pd.DataFrame:
        """Enriches movies with tags and TMDB links from ml-latest-small if available."""
        tags_file = self.data_dir / "ml-latest-small" / "tags.csv"
        links_file = self.data_dir / "ml-latest-small" / "links.csv"
        latest_movies_file = self.data_dir / "ml-latest-small" / "movies.csv"

        if not (tags_file.exists() and latest_movies_file.exists()):
            return movies

        try:
            tags_df = pd.read_csv(tags_file)
            latest_movies = pd.read_csv(latest_movies_file)

            # Clean and normalize titles for matching between ml-1m and ml-latest-small
            def norm_title(s: str) -> str:
                s = str(s).lower()
                s = re.sub(r"[^\w\s]", "", s)
                return re.sub(r"\s+", " ", s).strip()

            latest_movies["norm_title"] = latest_movies["title"].apply(norm_title)
            # Group tags by movieId
            movie_tags = (
                tags_df.groupby("movieId")["tag"]
                .apply(lambda ts: " ".join(ts.astype(str).str.lower().unique()[:15]))
                .reset_index()
            )
            latest_with_tags = latest_movies.merge(movie_tags, on="movieId", how="left")

            # Merge links for tmdbId
            if links_file.exists():
                links_df = pd.read_csv(links_file)
                latest_with_tags = latest_with_tags.merge(links_df[["movieId", "tmdbId", "imdbId"]], on="movieId", how="left")

            # Match with ml-1m movies by normalized title
            movies["norm_title"] = movies["title"].apply(norm_title)

            # Drop duplicates in lookup table
            tag_lookup = latest_with_tags.dropna(subset=["norm_title"]).drop_duplicates("norm_title")
            cols_to_merge = ["norm_title", "tag"]
            if "tmdbId" in tag_lookup.columns:
                cols_to_merge.append("tmdbId")
            if "imdbId" in tag_lookup.columns:
                cols_to_merge.append("imdbId")

            merged = movies.merge(tag_lookup[cols_to_merge], on="norm_title", how="left")
            merged["tags"] = merged["tag"].fillna("")
            merged.drop(columns=["norm_title", "tag"], inplace=True)

            # Update metadata soup with rich tags (weight genres 3x and user tags 2x)
            genre_tokens = merged["genres"].str.replace("|", " ", regex=False)
            tag_tokens = merged["tags"].apply(lambda t: (str(t).strip() + " ") if str(t).strip() else "")
            merged["metadata_soup"] = (
                (genre_tokens + " ") * 3
                + (tag_tokens * 2)
                + merged["clean_title"]
                + " "
                + merged["decade"]
            )
            return merged

        except Exception as e:
            print(f"[DataPreprocessor] Warning: tag enrichment encountered an issue: {e}")
            if "norm_title" in movies.columns:
                movies.drop(columns=["norm_title"], inplace=True)
            return movies
