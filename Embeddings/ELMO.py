import os
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from allennlp.commands.elmo import ElmoEmbedder

# === Configuration ===
DATASETS = ["df1", "df3", "df4"]
SPLITS = ["train", "test"]
TEXT_COLUMN = "Text"

# === Load Elmo model ===
elmo = ElmoEmbedder()

# === Feature extraction ===
def extract_features(text):
    # Input must be a tokenized list
    if isinstance(text, str):
        tokens = text.split()
    elif isinstance(text, list):
        tokens = text
    else:
        raise ValueError("Text must be a string or list of tokens.")
    embeddings = elmo.embed_sentence(tokens)
    return embeddings.mean(axis=0)  # shape: (seq_len, 1024)

# === Run for a single dataset/split ===
def run_batch(dataset_name, split):
    df_path = f"data/{dataset_name}_{split}.csv"
    out_path = f"outputs/ELMO_{dataset_name}_{split}.npz"

    df = pd.read_csv(df_path)
    df = df.dropna(subset=[TEXT_COLUMN])
    texts = df[TEXT_COLUMN].astype(str).tolist()

    print(f">>> Extracting ELMo: {dataset_name}_{split} | Samples: {len(texts)}")
    feature_list = []
    for text in tqdm(texts):
        features = extract_features(text)
        feature_list.append(torch.tensor(features).mean(dim=0).unsqueeze(0))  # mean over seq_len

    feature_tensor = torch.cat(feature_list, dim=0).cpu().numpy()
    np.savez_compressed(out_path, feature_tensor)
    print(f"Saved: {out_path} | Shape: {feature_tensor.shape}")

# === Loop through all datasets ===
for dataset in DATASETS:
    for split in SPLITS:
        run_batch(dataset, split)
