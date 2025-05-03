import os
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from compute_instance_hardness import load_dataset_csv


def load_features(dataset_names, feature_names, data_types, feature_dir):
    feature_data = {}
    for dataset_name in dataset_names:
        for feature_name in feature_names:
            for data_type in data_types:
                feature_file = os.path.join(feature_dir, f"{feature_name}_{dataset_name}_{data_type}.npz")
                feature_key = f"{feature_name}_{dataset_name}_{data_type}"
                feature_data[feature_key] = np.load(feature_file)['arr_0']
    return feature_data


def load_train_hardness_matrix(dataset_name, hardness_dir):
    path = os.path.join(hardness_dir, f"New_{dataset_name}_hardness_matrix_train.npy")
    return np.load(path)


def compute_test_hardness_matrix(dataset_name, feature_names, loaded_features,
                                  train_labels, train_hardness_matrix, save_dir,
                                  k_neighbors=5):
    num_features = len(feature_names)
    feature_key_ref = f"{feature_names[0]}_{dataset_name}_test"
    num_instances_test = len(loaded_features[feature_key_ref])
    hardness_matrix_test = np.zeros((num_instances_test, num_features))

    for i, feature_name in tqdm(enumerate(feature_names), total=num_features,
                                desc=f"Processing {dataset_name} Test"):
        key_train = f"{feature_name}_{dataset_name}_train"
        key_test = f"{feature_name}_{dataset_name}_test"

        X_train = loaded_features[key_train]
        X_test = loaded_features[key_test]
        y_train = train_labels

        nbrs = NearestNeighbors(n_neighbors=k_neighbors, algorithm='kd_tree').fit(X_train)

        for j in range(num_instances_test):
            _, indices = nbrs.kneighbors(X_test[j].reshape(1, -1))
            neighbors = indices[0]
            avg_hardness = np.mean(train_hardness_matrix[neighbors, i])
            hardness_matrix_test[j, i] = avg_hardness

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"New_{dataset_name}_hardness_matrix_test.npy")
    np.save(save_path, hardness_matrix_test)
    return hardness_matrix_test


def process_multiple_datasets(base_dir, dataset_names, feature_names, data_types, k_neighbors=5):
    dataset_dir = os.path.join(base_dir, "Dataset")
    features_dir = os.path.join(base_dir, "Features")
    ih_dir = os.path.join(base_dir, "IH")

    features = load_features(dataset_names, feature_names, data_types, features_dir)

    for dataset_name in dataset_names:
        print(f"\n=== Processing {dataset_name} ===")
        df_train = load_dataset_csv(dataset_dir, f"{dataset_name}_train")
        train_labels = df_train['Label'].values
        train_hardness = load_train_hardness_matrix(dataset_name, ih_dir)

        compute_test_hardness_matrix(
            dataset_name=dataset_name,
            feature_names=feature_names,
            loaded_features=features,
            train_labels=train_labels,
            train_hardness_matrix=train_hardness,
            save_dir=ih_dir,
            k_neighbors=k_neighbors
        )


# ===== Example Usage =====
if __name__ == "__main__":
    BASE_DIR = "/your/local/NLP_GNN"
    DATASET_NAMES = ["df1", "df2", "df3", "df4"]
    FEATURE_NAMES = [
        "W2V", "Glove", "Fasttext", "ELMO", "BERT", "DistilBERT", "ALBERT",
        "RoBERTa", "BART", "ELECTRA", "XLNET", "LLAMA", "Falcon", "LLAMA3", "MISTRAL"
    ]
    DATA_TYPES = ["train", "test"]

    process_multiple_datasets(BASE_DIR, DATASET_NAMES, FEATURE_NAMES, DATA_TYPES)
