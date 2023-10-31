import numpy as np
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
k_neighbors = 5
num_features = len(feature_names)


dataset_names = ["df1"]
feature_names = ["W2V", "Glove", "Fasttext","ELMO", "BERT", "DistilBERT", "ALBERT", "RoBERTa", "BART", "ELECTRA", "XLNET","LLAMA", "Falcon"]
data_types = ["train","test"]

def load_features(dataset_names, feature_names, data_types):
    feature_data = {}
    for dataset_name in dataset_names:
        for feature_name in feature_names:
            for data_type in data_types:
                feature_file = f"/content/gdrive/MyDrive/NLP_GNN/Features/{feature_name}_{dataset_name}_{data_type}.npz"
                feature_key = f"{feature_name}_{dataset_name}_{data_type}"
                feature_data[feature_key] = np.load(feature_file)['arr_0']

    return feature_data

loaded_features = load_features(dataset_names, feature_names, data_types)
len(loaded_features.keys())


num_instances_test = len(loaded_features['W2V_df1_test'])
hardness_matrix_test = np.zeros((num_instances_test, num_features))


for i, feature_name in tqdm(enumerate(feature_names), total=num_features, desc="Processing Features"):
    feature_key_train = f"{feature_name}_df1_train"
    X_train = loaded_features[feature_key_train]
    y_train = df1_train_class['Label'].values

    nbrs = NearestNeighbors(n_neighbors=k_neighbors, algorithm='kd_tree').fit(X_train)


    for j in range(num_instances_test):
        feature_key_test = f"{feature_name}_df1_test"
        _, indices = nbrs.kneighbors(loaded_features[feature_key_test][j].reshape(1, -1))
        neighbors_train = indices[0]


        avg_instance_hardness = np.mean(df1_hardness_matrix_train[neighbors_train, i])

        hardness_matrix_test[j, i] = avg_instance_hardness

np.save('/content/gdrive/MyDrive/NLP_GNN/New_df1_hardness_matrix_test.npy', hardness_matrix_test)
