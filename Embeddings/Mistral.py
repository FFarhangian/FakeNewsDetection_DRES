import os
from extract_embeddings import run_extraction

DATASETS = ["df1", "df3", "df4"]
SPLITS = ["train", "test"]
MODEL_PATH = "models/mistral"

for dataset in DATASETS:
    for split in SPLITS:
        csv_path = f"data/{dataset}_{split}.csv"
        out_file = f"outputs/MISTRAL_{dataset}_{split}.npz"
        print(f"\n>>> Processing: {csv_path} with MISTRAL model")
        run_extraction(
            df_path=csv_path,
            model_path=MODEL_PATH,
            out_file=out_file,
            pooling="mean",
            text_col="Text"
        )
