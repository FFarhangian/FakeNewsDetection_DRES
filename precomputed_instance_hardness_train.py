import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from instance_hardness import kdn_score 

# ========== Step 1: Constants ========== #

FEATURE_NAMES = [
    "W2V", "Glove", "Fasttext", "ELMO", "BERT", "DistilBERT", "ALBERT",
    "RoBERTa", "BART", "ELECTRA", "XLNET", "LLAMA", "Falcon", "LLAMA3", "MISTRAL"
]

DATASET_NAMES = ["df1", "df2", "df3", "df4"]
DATA_TYPES = ["train", "test"]

# Set your local paths here
BASE_DIR = "/your/local/path/to/NLP_GNN"
DATASET_DIR = os.path.join(BASE_DIR, "Dataset")
FEATURES_DIR = os.path.join(BASE_DIR, "Features")
IH_SAVE_DIR = os.path.join(BASE_DIR, "IH")

# ========== Step 2: Utility Functions ========== #

def load_dataset_csv(dataset_dir, name):
    return pd.read_csv(os.path.join(dataset_dir, f"{name}_class.csv"))

def load_features(dataset_names, feature_names, data_types, feature_dir):
    feature_data = {}
    for dataset in dataset_names:
        for feature in feature_names:
            for dtype in data_types:
                key = f"{feature}_{dataset}_{dtype}"
                path = os.path.join(feature_dir, f"{key}.npz")
                feature_data[key] = np.load(path)['arr_0']
    return feature_data

def compute_hardness_matrix(df_train, dataset_name, feature_names, features_dict, save_path, k=5):
    num_instances = len(df_train)
    num_features = len(feature_names)
    matrix = np.zeros((num_instances, num_features))

    for i, feature in tqdm(enumerate(feature_names), total=num_features, desc=f"Processing {dataset_name}"):
        key = f"{feature}_{dataset_name}_train"
        X = features_dict[key]
        y = df_train["Label"].values
        scores, _ = kdn_score(X, y, k)
        matrix[:, i] = scores

    os.makedirs(save_path, exist_ok=True)
    np.save(os.path.join(save_path, f"New_{dataset_name}_hardness_matrix_train.npy"), matrix)
    return matrix

# ========== Step 3: Main Execution ========== #

def main():
    train_dfs = {
        name: load_dataset_csv(DATASET_DIR, f"{name}_train") for name in DATASET_NAMES
    }

    features = load_features(DATASET_NAMES, FEATURE_NAMES, DATA_TYPES, FEATURES_DIR)

    for dataset in DATASET_NAMES:
        compute_hardness_matrix(train_dfs[dataset], dataset, FEATURE_NAMES, features, IH_SAVE_DIR)

if __name__ == "__main__":
    main()
