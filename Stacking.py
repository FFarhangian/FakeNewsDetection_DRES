import os
import glob
import numpy as np
from tqdm import tqdm

sorted_hardness_matrix = np.load('/content/gdrive/MyDrive/NLP_GNN/df1_sorted_feature_matrix_train.npy', allow_pickle=True)


final_record_index_matrices = []

for idx, record in enumerate(tqdm(sorted_hardness_matrix)):
    print (idx,record)
    path = "/content/gdrive/MyDrive/Ensemble_Learning/probs"
    files = os.path.join(path, f'df1_*_{record}_train_probs.npz')
    file_list = glob.glob(files)

    file_names = [os.path.basename(file).split('_train_probs.npz')[0] for file in file_list]


    record_probs = []

    for file in file_list:
        data = np.load(file)

        if 'probs' in data:
            record_probs.append(data['probs'])
        elif 'probabilities' in data:
            record_probs.append(data['probabilities'])
        else:
            print(f"Warning: No appropriate key found in {file}")
    

    if record_probs:
        record_prob_matrix = np.hstack(record_probs)
        

        record_index_matrix = record_prob_matrix[idx:idx+1]
        
        final_record_index_matrices.append(record_index_matrix)


if final_record_index_matrices:
    final_record_index_matrix = np.vstack(final_record_index_matrices)


print(f"Shape of the final matrix: {final_record_index_matrix.shape}")

np.save('/content/gdrive/MyDrive/NLP_GNN/df1_train_final_record_index_matrix.npy', final_record_index_matrix)

