"""
FastAPI app that uses collaborative filtering + NCF model
to recommend insurance policies.
"""

from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.recommender import build_user_item_matrix, compute_similarity, recommend_policies

DATA_PATH = Path("data/user_policy_ratings.csv")
MODEL_PATH = Path("models/ncf_model.keras")

app = FastAPI(title="Insurance Recommendation Engine")

ratings_df = pd.read_csv(DATA_PATH)
user_item = build_user_item_matrix(ratings_df)
similarity = compute_similarity(user_item)

if MODEL_PATH.exists():
    ncf_model = tf.keras.models.load_model(MODEL_PATH)
else:
    ncf_model = None


class RecommendationRequest(BaseModel):
    user_id: int = Field(..., ge=1)
    top_n: int = Field(5, ge=1, le=20)


@app.get("/")
async def root():
    return {"service": "insurance-recommendation-engine", "status": "ok"}


@app.post("/recommend")
async def recommend(req: RecommendationRequest):
    base_recs = recommend_policies(
        user_item, similarity, req.user_id, top_n=req.top_n
    )

    ncf_scores = {}
    if ncf_model is not None and base_recs:
        # Map to internal indices
        user_ids = ratings_df["user_id"].unique()
        policy_ids = ratings_df["policy_id"].unique()

        user_id_to_idx = {uid: i for i, uid in enumerate(sorted(user_ids))}
        policy_id_to_idx = {pid: i for i, pid in enumerate(sorted(policy_ids))}

        if req.user_id in user_id_to_idx:
            u_idx = user_id_to_idx[req.user_id]
            for pid in base_recs:
                if pid in policy_id_to_idx:
                    p_idx = policy_id_to_idx[pid]
                    pred = ncf_model.predict(
                        {"user": np.array([u_idx]), "policy": np.array([p_idx])},
                        verbose=0,
                    )[0][0]
                    ncf_scores[pid] = float(pred)

    ranked = (
        sorted(ncf_scores.items(), key=lambda x: x[1], reverse=True)
        if ncf_scores
        else [(pid, None) for pid in base_recs]
    )

    return {
        "user_id": req.user_id,
        "recommendations": [
            {"policy_id": int(pid), "score": score} for pid, score in ranked
        ],
    }
