# build_index.py
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

INPUT = Path("training_examples.csv")
OUT = Path("examples_index.pkl")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    if not INPUT.exists():
        raise FileNotFoundError(
            f"Could not find {INPUT.resolve()}.\n"
            "Create training_examples.csv with columns: description,family,category1"
        )

    # Load and validate
    df = pd.read_csv(INPUT, dtype=str).fillna("")
    needed = {"description", "family", "category1"}
    missing = needed - set(c.strip().lower() for c in df.columns)
    if missing:
        raise ValueError(
            "training_examples.csv must have columns exactly: description,family,category1"
        )

    # Normalize column names in case of casing/whitespace issues
    df.columns = [c.strip().lower() for c in df.columns]
    df = df[["description", "family", "category1"]]

    # Drop empty and duplicates
    df["description"] = df["description"].astype(str).str.strip()
    df["family"] = df["family"].astype(str).str.strip()
    df["category1"] = df["category1"].astype(str).str.strip()
    df = df[(df["description"] != "") & (df["family"] != "") & (df["category1"] != "")]
    df = df.drop_duplicates(subset=["description", "family", "category1"]).reset_index(drop=True)

    if df.empty:
        raise ValueError("training_examples.csv is empty after cleaning.")

    # Load embedding model (downloads on first run, then cached)
    model = SentenceTransformer(MODEL_NAME)

    # Compute normalized embeddings for similarity search
    texts = df["description"].tolist()
    emb = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    emb = np.asarray(emb, dtype=np.float32)

    payload = {
        "model_name": MODEL_NAME,
        "embeddings": emb,  # shape: (N, D) float32, L2-normalized
        "descriptions": df["description"].tolist(),
        "families": df["family"].tolist(),
        "categories": df["category1"].tolist(),
    }

    with open(OUT, "wb") as f:
        pickle.dump(payload, f)

    print(f"Saved index: {OUT} with {len(df)} example(s).")


if __name__ == "__main__":
    main()
