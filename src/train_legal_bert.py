"""Fase 2 - Fine-tuning de Legal-BERT para classificação de cláusulas.

O pipeline é deliberadamente configurável para execução em CPU. Por padrão usa
um subconjunto pequeno e uma época para smoke test; o treinamento completo é
ativado com ``--max-samples 0 --epochs 3`` e pode levar bastante tempo sem GPU.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
DEFAULT_MODEL = "nlpaueb/legal-bert-base-uncased"
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data/raw/cuad_clauses.csv"
REPORT_DIR = BASE_DIR / "reports"
MODEL_DIR = BASE_DIR / "data/processed/legal_bert"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=200,
                        help="0 usa todas as cláusulas")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--output-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    return parser.parse_args(argv)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def prepare_data(df: pd.DataFrame, max_samples: int, seed: int = RANDOM_STATE):
    data = df.dropna(subset=["clause", "label"])[["clause", "label"]].copy()
    if max_samples:
        labels_in_full_data = data["label"].nunique()
        minimum_for_split = 2 * labels_in_full_data
        if max_samples < minimum_for_split:
            raise ValueError(
                f"--max-samples deve ser >= {minimum_for_split} para manter "
                "duas amostras por classe (ou use --max-samples 0)."
            )
        # Garante pelo menos duas observações por classe antes de completar
        # a amostra aleatoriamente; isso torna smoke tests pequenos robustos.
        mandatory = data.groupby("label", group_keys=False).sample(
            n=2, random_state=seed
        )
        remaining = data.drop(index=mandatory.index)
        extra = remaining.sample(
            n=max_samples - len(mandatory), random_state=seed
        )
        data = pd.concat([mandatory, extra]).sample(frac=1, random_state=seed)
    labels = sorted(data["label"].unique())
    label_to_id = {label: i for i, label in enumerate(labels)}
    data["label_id"] = data["label"].map(label_to_id)
    test_count = max(len(labels), math.ceil(0.2 * len(data)))
    train, test = train_test_split(
        data, test_size=test_count, random_state=seed, stratify=data["label_id"]
    )
    return train.reset_index(drop=True), test.reset_index(drop=True), label_to_id


def main(argv=None):
    args = parse_args(argv)
    set_seed(args.seed)
    REPORT_DIR.mkdir(exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Imports pesados ficam dentro da execução para permitir testar os helpers
    # sem instalar o stack de deep learning.
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    df = pd.read_csv(DATA_PATH)
    train_df, test_df, label_to_id = prepare_data(df, args.max_samples, args.seed)
    id_to_label = {v: k for k, v in label_to_id.items()}
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    class ClauseDataset(Dataset):
        def __init__(self, frame):
            self.texts = frame["clause"].tolist()
            self.labels = frame["label_id"].astype(int).tolist()

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, index):
            encoded = tokenizer(
                self.texts[index], truncation=True, padding="max_length",
                max_length=args.max_length, return_tensors="pt"
            )
            return {k: v.squeeze(0) for k, v in encoded.items()} | {
                "labels": torch.tensor(self.labels[index], dtype=torch.long)
            }

    train_loader = DataLoader(ClauseDataset(train_df), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(ClauseDataset(test_df), batch_size=args.batch_size)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=len(label_to_id), ignore_mismatched_sizes=True
    )
    model.config.id2label = id_to_label
    model.config.label2id = label_to_id
    device = torch.device("cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    for epoch in range(args.epochs):
        model.train()
        losses = []
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        print(f"epoch={epoch + 1}/{args.epochs} train_loss={np.mean(losses):.4f}")

    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in test_loader:
            labels = batch.pop("labels").numpy().tolist()
            outputs = model(**{k: v.to(device) for k, v in batch.items()})
            predictions = outputs.logits.argmax(dim=-1).cpu().numpy().tolist()
            y_true.extend(labels)
            y_pred.extend(predictions)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "n_train": len(train_df), "n_test": len(test_df),
        "n_classes": len(label_to_id), "model_name": args.model_name,
        "epochs": args.epochs, "max_samples": args.max_samples,
        "device": str(device),
    }
    with open(REPORT_DIR / "legal_bert_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    main()
