import os
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from transformers import BartModel, BartTokenizer

# === CONFIGURATION ===
DATASETS = ["df1", "df3", "df4"]
SPLITS = ["train", "test"]
TEXT_COLUMN = "Text"
MODEL_NAME = "facebook/bart-base"

# === Load model and tokenizer ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BartModel.from_pretrained(MODEL_NAME, output_hidden_states=True).to(device)
tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)
model.eval()

# === Helper functions ===
def preprocess(text):
    return tokenizer(text, truncation=True, padding="max_length", max_length=512, return_tensors="pt").to(device)

def extract_features(text):
    inputs = preprocess(text)
    with torch.no_grad():
        outputs = model(**inputs)
    last_tokens = outputs.last_hidden_state[:, -4:, :]  # Last 4 tokens
    return last_tokens.mean(dim=1)

def run_batch(dataset_name, split):
    df_path = f"data/{dataset_name}_{split}.csv"
    out_path = f"outputs/BART_{dataset_name}_{split}.npz"

    df = pd.read_csv(df_path)
    df = df.dropna(subset=[TEXT_COLUMN])
    texts = df[TEXT_COLUMN].astype(str).tolist()

    print(f">>> Processing: {dataset_name}_{split} with BART | Samples: {len(texts)}")
    features = [extract_features(text) for text in tqdm(texts)]
    feature_matrix = torch.cat(features, dim=0).cpu().numpy()

    np.savez_compressed(out_path, feature_matrix)
    print(f"Saved to {out_path} | Shape: {feature_matrix.shape}")

# === Run extraction ===
for dataset in DATASETS:
    for split in SPLITS:
        run_batch(dataset, split)
