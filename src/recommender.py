"""
User-based collaborative filtering recommender for insurance policies.
"""

from typing import List

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def build_user_item_matrix(ratings: pd.DataFrame) -> pd.DataFrame:
    ui = ratings.pivot_table(
        index="user_id", columns="policy_id", values="rating"
    ).fillna(0.0)
    return ui


def compute_similarity(user_item: pd.DataFrame) -> pd.DataFrame:
    sim = cosine_similarity(user_item.values)
    return pd.DataFrame(sim, index=user_item.index, columns=user_item.index)


def recommend_policies(
    user_item: pd.DataFrame,
    similarity: pd.DataFrame,
    user_id: int,
    top_n: int = 5,
) -> List[int]:
    if user_id not in user_item.index:
        return []

    user_sim = similarity.loc[user_id].sort_values(ascending=False)
    neighbors = user_sim.index[1:11]  # skip self

    already_rated = set(
        user_item.loc[user_id][user_item.loc[user_id] > 0].index
    )

    scores = {}

    for other in neighbors:
        weight = user_sim[other]
        other_ratings = user_item.loc[other]
        for policy_id, rating in other_ratings.items():
            if rating <= 0 or policy_id in already_rated:
                continue
            scores[policy_id] = scores.get(policy_id, 0.0) + weight * rating

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [int(pid) for pid, _ in ranked[:top_n]]
