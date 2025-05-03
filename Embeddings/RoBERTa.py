import os
import pandas as pd
import torch
import numpy as np
from transformers import RobertaModel, RobertaTokenizer
from tqdm import tqdm

# === Configuration ===
DATASETS = ["df1", "df3", "df4"]
SPLITS = ["train", "test"]
MODEL_NAME = "roberta-base"
TEXT_COLUMN = "Text"

# === Load Model & Tokenizer ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RobertaModel.from_pretrained(MODEL_NAME, output_hidden_states=True).to(device)
tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
model.eval()

# === Helper Functions ===
def preprocess(text):
    return tokenizer(text, truncation=True, padding="max_length", max_length=512, return_tensors="pt").to(device)

def extract_features(text):
    inputs = preprocess(text)
    with torch.no_grad():
        output = model(**inputs)
    last_tokens = output.last_hidden_state[:, -4:, :]
    return last_tokens.mean(dim=1)

def run_batch(dataset_name, split):
    df_path = f"data/{dataset_name}_{split}.csv"
    out_path = f"outputs/RoBERTa_{dataset_name}_{split}.npz"

    df = pd.read_csv(df_path)
    df = df.dropna(subset=[TEXT_COLUMN])
    texts = df[TEXT_COLUMN].astype(str).tolist()

    print(f">>> Extracting: {dataset_name}_{split} | Samples: {len(texts)}")
    embeddings = []
    for text in tqdm(texts):
        features = extract_features(text)
        embeddings.append(features)

    feature_tensor = torch.cat(embeddings, dim=0).cpu().numpy()
    np.savez_compressed(out_path, feature_tensor)
    print(f"Saved to {out_path} | Shape: {feature_tensor.shape}")

# === Run all extractions ===
for dataset in DATASETS:
    for split in SPLITS:
        run_batch(dataset, split)
