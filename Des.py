import os
import glob
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from deslib2.des.knora_e import KNORAE
from deslib2.des.meta_des import METADES
from deslib2.des.des_p import DESP
from deslib2.static.oracle import Oracle

# Constants
DATASET_NAMES = ["df1", "df3", "df4"]
FEATURE_NAMES = ["ALBERT", "BART", "BERT", "DistilBERT", "ELECTRA", "ELMO", "Falcon", "Fasttext", "Glove", "LLAMA",
                 "RoBERTa", "W2V", "XLNET", "LLAMA3", "MISTRAL"]
DATA_TYPES = ["train", "test"]
ALGORITHMS = ["AdaBoost", "BiLSTM", "CNN", "KNN", "LR", "MLP", "NB", "RF", "SVM", "XGBoost"]
PATH_NAME = 'path'


def load_features(dataset_names, feature_names, data_types):
    feature_data = {}
    for dataset in dataset_names:
        for feature in feature_names:
            for dtype in data_types:
                fname = f"Features/{feature}_{dataset}_{dtype}.npz"
                key = f"{feature}_{dataset}_{dtype}"
                feature_data[key] = np.load(fname)['arr_0']
    return feature_data


def process_preds(dataset, data_type, feature_name, algorithms, path_name):
    path = os.path.join(path_name, "preds", dataset)
    combined_df_list = []
    for alg in algorithms:
        file = os.path.join(path, f"{dataset}_{alg}_{feature_name}_{data_type}_preds.npz")
        if os.path.exists(file):
            data = np.load(file)
            col_name = os.path.basename(file).split(f'_{data_type}_preds.npz')[0]
            preds = data.get('preds') or data.get('predictions')
            if preds is not None:
                df = pd.DataFrame({col_name: preds})
                combined_df_list.append(df)
    return combined_df_list


def process_probs(dataset, data_type, feature_name, path_name):
    path = os.path.join(path_name, "probs", dataset)
    files = glob.glob(os.path.join(path, f"{dataset}_*_{feature_name}_{data_type}_probs.npz"))
    df = pd.DataFrame()
    for file in files:
        data = np.load(file)
        col_name = os.path.basename(file).split(f'_{data_type}_probs.npz')[0]
        probs = data.get('probs') or data.get('probabilities')
        if probs is not None:
            for c in range(probs.shape[1]):
                df[f"{col_name}_{c}"] = probs[:, c]
    return df


def save_selected_classifiers(clf, feature_name, method_name, algorithms):
    selected_dir = "selected_classifiers"
    os.makedirs(selected_dir, exist_ok=True)
    path = os.path.join(selected_dir, f"{feature_name}_{method_name}_selected.csv")

    if hasattr(clf, "supports_"):
        selected = clf.supports_.astype(int)
        pd.DataFrame(selected, columns=algorithms).to_csv(path, index=False)
    elif hasattr(clf, "selected_classifiers_"):
        selected = clf.selected_classifiers_.astype(int)
        pd.DataFrame(selected, columns=algorithms).to_csv(path, index=False)
    else:
        print(f"[!] Cannot extract selected classifiers for {method_name}")


def evaluate_feature(feature_name, df4_train_class, df4_test_class, loaded_features):
    X_train = loaded_features[f"{feature_name}_df4_train"]
    X_test = loaded_features[f"{feature_name}_df4_test"]
    y_train = df4_train_class.to_numpy().flatten()
    y_test = df4_test_class.to_numpy().flatten()

    preds_train = [df.to_numpy().flatten() for df in process_preds("df4", "train", feature_name, ALGORITHMS, PATH_NAME)]
    preds_test = [df.to_numpy().flatten() for df in process_preds("df4", "test", feature_name, ALGORITHMS, PATH_NAME)]
    preds_test = np.transpose(np.array(preds_test))

    classifiers = {
        "KNORAE": KNORAE(k=5, alg_pool=ALGORITHMS),
        "DESP": DESP(alg_pool=ALGORITHMS),
        "METADES": METADES(alg_pool=ALGORITHMS),
        "Oracle": Oracle(alg_pool=ALGORITHMS)
    }

    matrices = {}
    accuracies = {}
    pred_dict = {}

    for name, clf in classifiers.items():
        clf.fit(X_train, y_train, preds_train)
        if name == "Oracle":
            pred_test_clf = clf.predict(X_test, y_test, preds_test)
        else:
            pred_test_clf = clf.predict(X_test, preds_test)

        matrices[name] = pred_test_clf
        accuracies[name] = accuracy_score(y_test, pred_test_clf)
        pred_dict[name] = pred_test_clf

        print(f"[✓] {name} on {feature_name}: {accuracies[name]:.4f}")
        save_selected_classifiers(clf, feature_name, name, ALGORITHMS)

    return matrices, accuracies, pred_dict


def main():
    os.makedirs("predictions", exist_ok=True)
    os.makedirs("accuracy_reports", exist_ok=True)

    loaded_features = load_features(DATASET_NAMES, FEATURE_NAMES, DATA_TYPES)
    df4_train_class = pd.read_csv(PATH_NAME + 'Dataset/df4_train_class.csv')
    df4_test_class = pd.read_csv(PATH_NAME + 'Dataset/df4_test_class.csv')

    final_matrices = {name: [] for name in ["KNORAE", "DESP", "METADES", "Oracle"]}
    final_accuracies = {name: [] for name in ["KNORAE", "DESP", "METADES", "Oracle"]}

    for feature_name in FEATURE_NAMES:
        matrices, accuracies, preds = evaluate_feature(feature_name, df4_train_class, df4_test_class, loaded_features)
        for method in matrices:
            final_matrices[method].append(matrices[method])
            final_accuracies[method].append((feature_name, accuracies[method]))
            pd.DataFrame(preds[method], columns=[f"{method}_preds"]).to_csv(f"predictions/df4_{feature_name}_{method}_predictions.csv", index=False)

    for method, matrix_list in final_matrices.items():
        mat = np.column_stack(matrix_list)
        pd.DataFrame(mat, columns=FEATURE_NAMES).to_csv(f"df4_{method}_matrix.csv", index=False)

    for method, acc_list in final_accuracies.items():
        pd.DataFrame(acc_list, columns=["Feature", "Accuracy"]).to_csv(f"accuracy_reports/df4_{method}_accuracies.csv", index=False)


if __name__ == "__main__":
    main()
