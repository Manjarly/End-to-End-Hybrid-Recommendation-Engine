"""
Automated downloader and cache manager for MovieLens datasets.
Supports MovieLens 1M (1,000,209 ratings), MovieLens Latest Small, and MovieLens 100K.
"""

import os
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, Optional


DATASET_URLS: Dict[str, str] = {
    "ml-1m": "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
    "ml-latest-small": "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip",
    "ml-100k": "https://files.grouplens.org/datasets/movielens/ml-100k.zip",
}


class DatasetDownloader:
    """Manages downloading, caching, and unpacking MovieLens datasets."""

    def __init__(self, cache_dir: Optional[str] = None):
        if cache_dir is None:
            # Default to data/ directory in project root
            project_root = Path(__file__).resolve().parent.parent.parent
            self.cache_dir = project_root / "data"
        else:
            self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def download_and_extract(self, dataset_name: str = "ml-1m", force: bool = False) -> Path:
        """
        Ensures the dataset is downloaded and extracted.
        Returns the path to the extracted dataset directory.
        """
        if dataset_name not in DATASET_URLS:
            raise ValueError(
                f"Unknown dataset '{dataset_name}'. Available: {list(DATASET_URLS.keys())}"
            )

        target_dir = self.cache_dir / dataset_name
        zip_path = self.cache_dir / f"{dataset_name}.zip"

        # Check if already extracted
        if target_dir.exists() and not force:
            # Check key files
            if dataset_name == "ml-1m" and (target_dir / "ratings.dat").exists():
                print(f"[DatasetDownloader] '{dataset_name}' already extracted at: {target_dir}")
                return target_dir
            elif dataset_name == "ml-latest-small" and (target_dir / "ratings.csv").exists():
                print(f"[DatasetDownloader] '{dataset_name}' already extracted at: {target_dir}")
                return target_dir
            elif dataset_name == "ml-100k" and (target_dir / "u.data").exists():
                print(f"[DatasetDownloader] '{dataset_name}' already extracted at: {target_dir}")
                return target_dir

        url = DATASET_URLS[dataset_name]
        print(f"[DatasetDownloader] Downloading {dataset_name} from {url}...")

        headers = {"User-Agent": "Mozilla/5.0 (HybridRecommender/2.0)"}
        req = urllib.request.Request(url, headers=headers)

        with urllib.request.urlopen(req, timeout=30) as response, open(zip_path, "wb") as out_file:
            content_length = response.headers.get("Content-Length")
            total_size = int(content_length) if content_length else 0
            downloaded = 0
            block_size = 64 * 1024

            while True:
                buffer = response.read(block_size)
                if not buffer:
                    break
                downloaded += len(buffer)
                out_file.write(buffer)
                if total_size > 0:
                    pct = (downloaded / total_size) * 100
                    print(f"\r  Progress: {downloaded / (1024*1024):.1f}MB / {total_size / (1024*1024):.1f}MB ({pct:.1f}%)", end="")

        print(f"\n[DatasetDownloader] Download completed: {zip_path}")
        print(f"[DatasetDownloader] Extracting into {self.cache_dir}...")

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(self.cache_dir)

        if zip_path.exists():
            zip_path.unlink()  # Clean up zip archive

        print(f"[DatasetDownloader] Extracted successfully to: {target_dir}")
        return target_dir

    def ensure_rich_dataset(self) -> Dict[str, Path]:
        """
        Ensures both MovieLens 1M (primary) and MovieLens Latest Small (for rich tags & TMDB links)
        are downloaded and ready for use.
        """
        ml_1m_path = self.download_and_extract("ml-1m")
        ml_tags_path = self.download_and_extract("ml-latest-small")
        return {
            "ml-1m": ml_1m_path,
            "ml-latest-small": ml_tags_path,
        }
