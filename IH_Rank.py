import numpy as np

dataset_names = ["df1", "df2", "df3", "df4"]
feature_names = ["W2V", "Glove", "Fasttext", "ELMO", "BERT", "DistilBERT", "ALBERT", "RoBERTa", "BART", "ELECTRA", "XLNET", "LLAMA", "Falcon"]

for dataset_name in dataset_names:
    hardness_matrix_train = np.load(f'/content/gdrive/MyDrive/NLP_GNN/New_{dataset_name}_hardness_matrix_train.npy')
    hardness_matrix_test = np.load(f'/content/gdrive/MyDrive/NLP_GNN/New_{dataset_name}_hardness_matrix_test.npy')

    sorted_feature_matrix_train = np.empty(hardness_matrix_train.shape[0], dtype=object)
    sorted_feature_matrix_test = np.empty(hardness_matrix_test.shape[0], dtype=object)

    for i in range(hardness_matrix_train.shape[0]):
        hardness_scores = hardness_matrix_train[i]

        sorted_features = [feature_name for _, feature_name in sorted(zip(hardness_scores, feature_names))]

        sorted_feature_matrix_train[i] = sorted_features[0]

    for i in range(hardness_matrix_test.shape[0]):
        hardness_scores = hardness_matrix_test[i]

        sorted_features = [feature_name for _, feature_name in sorted(zip(hardness_scores, feature_names))]

        sorted_feature_matrix_test[i] = sorted_features[0]

    np.save(f'/content/gdrive/MyDrive/NLP_GNN/{dataset_name}_sorted_feature_matrix_train.npy', sorted_feature_matrix_train)
    np.save(f'/content/gdrive/MyDrive/NLP_GNN/{dataset_name}_sorted_feature_matrix_test.npy', sorted_feature_matrix_test)

