from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer

from app.config import cfg
from app.data_loader import load_messages
from app.storage import load_parquet
from app.intents import embed_intents, zero_shot_filter


def main() -> None:
    load_dotenv()
    messages_df = load_messages(cfg.dataset_file)
    if messages_df.empty:
        print("No messages in dataset/result.json")
        return

    # Intents to test (edit as needed)
    intents = [
        "Identify bug reports and errors",
        "Feature requests and suggestions",
        "Positive feedback and appreciation",
    ]

    # Prefer precomputed embeddings if present and API key to embed intents
    emb_df = load_parquet(cfg.artifacts_dir / "embeddings.parquet")
    folder_id = os.getenv("FOLDER_ID")
    yc_key = os.getenv("API_KEY_EMBEDDER")
    yc_ready = bool(folder_id and yc_key)

    if emb_df is not None and yc_ready:
        print("Using Yandex Cloud embeddings for messages and intents")
        msg_feat_cols = [c for c in emb_df.columns if c.startswith("e")]
        message_embeddings = emb_df[msg_feat_cols].values
        intent_embeddings = embed_intents(intents)
    else:
        print("Falling back to TF-IDF for both messages and intents (no Yandex Cloud creds or no embeddings)")
        texts = messages_df["text"].astype(str).tolist()
        vectorizer = TfidfVectorizer(max_features=2048, ngram_range=(1, 2))
        X_msgs = vectorizer.fit_transform(texts).astype(np.float32)
        X_intents = vectorizer.transform(intents).astype(np.float32)
        message_embeddings = X_msgs.toarray()
        intent_embeddings = X_intents.toarray()

    results = zero_shot_filter(
        messages_df=messages_df,
        intents=intents,
        message_embeddings=message_embeddings,
        intent_embeddings=intent_embeddings,
        top_k=5,
        min_score=0.10,
    )

    print(f"Messages: {len(messages_df)}, Intents: {len(intents)}")
    for intent, df in results.items():
        print("\n== Intent:", intent)
        print("Top matches:")
        show = df[["date", "sender", "intent_score", "text"]].head(5).copy()
        # Trim text for readability
        show["text"] = show["text"].str.slice(0, 160)
        print(show.to_string(index=False))


if __name__ == "__main__":
    main()

