from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def compute_tfidf_features(
    df: pd.DataFrame,
    text_col: str = "text",
    id_col: str = "message_id",
    max_features: int = 2048,
    ngram_range: tuple[int, int] = (1, 2),
) -> pd.DataFrame:
    texts = df[text_col].astype(str).tolist()
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    X = vectorizer.fit_transform(texts)
    arr = X.toarray().astype(np.float32)
    feat_df = pd.DataFrame(arr)
    feat_df.columns = [f"e{i:04d}" for i in range(feat_df.shape[1])]
    feat_df[id_col] = df[id_col].astype(str).values
    return feat_df


