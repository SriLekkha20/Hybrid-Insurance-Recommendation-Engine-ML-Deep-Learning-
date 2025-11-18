"""
Train a simple Neural Collaborative Filtering model on user–policy ratings.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model

DATA_PATH = Path("data/user_policy_ratings.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True, parents=True)


def main():
    df = pd.read_csv(DATA_PATH)

    user_ids = df["user_id"].unique()
    policy_ids = df["policy_id"].unique()

    user_id_to_idx = {uid: i for i, uid in enumerate(sorted(user_ids))}
    policy_id_to_idx = {pid: i for i, pid in enumerate(sorted(policy_ids))}

    df["user_idx"] = df["user_id"].map(user_id_to_idx)
    df["policy_idx"] = df["policy_id"].map(policy_id_to_idx)

    num_users = len(user_id_to_idx)
    num_policies = len(policy_id_to_idx)

    X_users = df["user_idx"].values
    X_policies = df["policy_idx"].values
    y = df["rating"].values.astype("float32") / 5.0  # scale to [0,1]

    # Define a small NCF model
    user_input = layers.Input(shape=(), dtype="int32", name="user")
    item_input = layers.Input(shape=(), dtype="int32", name="policy")

    user_emb = layers.Embedding(num_users, 16)(user_input)
    item_emb = layers.Embedding(num_policies, 16)(item_input)

    x = layers.Concatenate()([user_emb, item_emb])
    x = layers.Flatten()(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dense(16, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid")(x)

    model = Model(inputs=[user_input, item_input], outputs=out)
    model.compile(optimizer="adam", loss="mse")

    model.fit(
        {"user": X_users, "policy": X_policies},
        y,
        batch_size=16,
        epochs=15,
        verbose=1,
    )

    model.save(MODEL_DIR / "ncf_model.keras")
    print("✅ Saved NCF model to models/ncf_model.keras")


if __name__ == "__main__":
    main()
