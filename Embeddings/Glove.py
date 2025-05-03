import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from zeugma.embeddings import EmbeddingTransformer

# === CONFIGURATION ===
DATASETS = ["df1", "df3", "df4"]
SPLITS = ["train", "test"]
TEXT_COLUMN = "Text"
MODEL_NAME = "glove"

# === Load embedding model ===
embedder = EmbeddingTransformer(MODEL_NAME)

# === Helper function ===
def run_batch(dataset_name, split):
    df_path = f"data/{dataset_name}_{split}.csv"
    out_path = f"outputs/Glove_{dataset_name}_{split}.npz"

    df = pd.read_csv(df_path)
    df = df.dropna(subset=[TEXT_COLUMN])
    texts = df[TEXT_COLUMN].astype(str).tolist()

    print(f">>> Processing: {dataset_name}_{split} with GloVe | Samples: {len(texts)}")
    if split == "train":
        feature_matrix = embedder.fit_transform(texts)
    else:
        feature_matrix = embedder.transform(texts)

    np.savez_compressed(out_path, feature_matrix)
    print(f"Saved to {out_path} | Shape: {feature_matrix.shape}")

# === Run extraction ===
for dataset in DATASETS:
    for split in SPLITS:
        run_batch(dataset, split)
