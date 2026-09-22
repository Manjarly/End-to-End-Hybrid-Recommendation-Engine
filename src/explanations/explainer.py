"""
Film Recommendation Curation and Note Generator.
Provides natural, editorial notes explaining why each title matches the viewer.
"""

from typing import Dict, List, Optional
from ..data.dataset import RecommendationDataset
from ..models.hybrid import HybridRecommender


class RecommendationExplainer:
    """Generates natural, readable curation notes for recommendations."""

    def __init__(self, dataset: RecommendationDataset, hybrid_model: HybridRecommender):
        self.dataset = dataset
        self.hybrid = hybrid_model

    def explain(
        self,
        user_id: int,
        movie_id: int,
        seed_movie_id: Optional[int] = None,
        score: float = 0.0,
        age_desc: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, str]:
        """
        Generates a human-friendly note for why movie_id fits the viewer.
        Free of technical jargon, emojis, or symbols.
        """
        title = self.dataset.movie_id_to_title.get(movie_id, "Selected Title")
        genres = self.dataset.movie_id_to_genres.get(movie_id, "")
        user_history = self.dataset.get_user_history(user_id, top_n=5)
        has_demo = bool(age_desc and str(age_desc).strip())

        # Case 1: Seed movie active (Viewer specified a reference film)
        if seed_movie_id is not None and seed_movie_id in self.dataset.movie_id_to_title:
            seed_title = self.dataset.movie_id_to_title[seed_movie_id]
            seed_genres = set(self.dataset.movie_id_to_genres.get(seed_movie_id, "").split("|"))
            item_genres = set(genres.split("|"))
            common = sorted(list((seed_genres & item_genres) - {"", "unknown"}))
            common_str = ", ".join(common).lower() if common else "storytelling and tone"

            demo_suffix = ""
            if has_demo and age_desc:
                demo_suffix = f" and highly favored by viewers aged {age_desc}"

            return {
                "badge": "Similar Tone",
                "badge_type": "tone",
                "title": title,
                "reason": f"Recommended for its thematic kinship with {seed_title}, particularly its focus on {common_str}{demo_suffix}.",
                "highlight": f"Echoes {seed_title}",
            }

        # Case 2: New viewer without prior ratings
        if not user_history:
            movie_row = self.dataset.movies[self.dataset.movies["movie_id"] == movie_id]
            b_rating = float(movie_row.iloc[0].get("bayesian_rating", 4.0)) if not movie_row.empty else 4.0
            r_count = int(movie_row.iloc[0].get("rating_count", 0)) if not movie_row.empty else 0

            if has_demo and age_desc:
                return {
                    "badge": "Age Group Favorite",
                    "badge_type": "favorite",
                    "title": title,
                    "reason": f"Consistently rated among the highest by viewers aged {age_desc}, averaging {b_rating:.1f} out of 5 across {r_count:,} audience reviews.",
                    "highlight": f"Top pick for age {age_desc}",
                }
            else:
                return {
                    "badge": "Viewer Favorite",
                    "badge_type": "favorite",
                    "title": title,
                    "reason": f"A widely acclaimed staple, averaging {b_rating:.1f} out of 5 from over {r_count:,} audience reviews.",
                    "highlight": "Widely acclaimed",
                }

        # Case 3: Viewer with established watch history
        item_genres = set(genres.split("|")) - {"", "unknown"}
        best_match_title = None
        best_match_overlap = 0

        for h in user_history:
            h_genres = set(h["genres"].split("|")) - {"", "unknown"}
            overlap = len(item_genres & h_genres)
            if overlap > best_match_overlap:
                best_match_overlap = overlap
                best_match_title = h["title"]

        if best_match_title and best_match_overlap >= 2:
            demo_suffix = f", with strong ratings among viewers aged {age_desc}" if (has_demo and age_desc) else ""
            return {
                "badge": "Curator Choice",
                "badge_type": "choice",
                "title": title,
                "reason": f"A natural companion for fans of {best_match_title}, sharing similar narrative depth and genre sensibilities{demo_suffix}.",
                "highlight": f"For fans of {best_match_title}",
            }
        else:
            if has_demo and age_desc:
                reason_text = f"Consistently praised by viewers in the {age_desc} age group who share your viewing preferences and appreciation for quality filmmaking."
            else:
                reason_text = "Consistently praised by viewers who share your viewing preferences and appreciation for quality filmmaking."
            return {
                "badge": "Audience Consensus",
                "badge_type": "consensus",
                "title": title,
                "reason": reason_text,
                "highlight": "Top pick for your taste profile",
            }
