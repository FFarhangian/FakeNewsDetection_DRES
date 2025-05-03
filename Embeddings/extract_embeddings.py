import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import numpy as np

def load_model_tokenizer(model_path, use_fp16=True):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if use_fp16 else torch.float32,
        output_hidden_states=True
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device), tokenizer, device

def preprocess(text, tokenizer, device):
    return tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    ).to(device)

def extract_features(text, model, tokenizer, device, pooling="mean"):
    model.eval()
    inputs = preprocess(text, tokenizer, device)
    with torch.no_grad():
        outputs = model(**inputs)
    last_hidden = outputs.hidden_states[-1]
    if pooling == "mean":
        mask = inputs["attention_mask"].unsqueeze(-1).expand(last_hidden.size()).float()
        summed = torch.sum(last_hidden * mask, 1)
        summed_mask = torch.clamp(mask.sum(1), min=1e-9)
        features = summed / summed_mask
    elif pooling == "cls":
        features = last_hidden[:, 0]
    else:
        raise ValueError("Unsupported pooling type: choose 'mean' or 'cls'")
    return F.normalize(features, p=2, dim=1)

def run_extraction(df_path, model_path, out_file, pooling="mean", text_col="Text"):
    df = pd.read_csv(df_path)
    df = df.dropna(subset=[text_col])
    df[text_col] = df[text_col].astype(str)
    model, tokenizer, device = load_model_tokenizer(model_path)
    features = []
    for text in df[text_col]:
        emb = extract_features(text, model, tokenizer, device, pooling)
        features.append(emb)
    features_tensor = torch.cat(features, dim=0)
    np.savez_compressed(out_file, features_tensor.cpu().numpy())
    print(f"Saved: {out_file} with shape {features_tensor.shape}")
