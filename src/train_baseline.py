"""Fase 1 - Baseline: TF-IDF + Regressão Logística para classificação de cláusulas (CUAD).

Nível técnico 1 do roadmap. Objetivo: piso de comparação interpretável antes do
fine-tuning de um encoder (Legal-BERT), que será feito na Fase 2.
"""
import pandas as pd
import numpy as np
import joblib
import json
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "cuad_clauses.csv")
MODEL_DIR = os.path.join(BASE_DIR, "data", "processed")
REPORT_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

RANDOM_STATE = 42


def main():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["clause", "label"])

    X = df["clause"].astype(str)
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Treino: {len(X_train)} | Teste: {len(X_test)}")

    vectorizer = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        stop_words="english",
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # class_weight='balanced' para compensar o desbalanceamento (~98x entre
    # a maior e a menor classe, identificado na EDA)
    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        C=5.0,
        random_state=RANDOM_STATE,
    )
    clf.fit(X_train_vec, y_train)

    y_pred = clf.predict(X_test_vec)

    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")

    print(f"\nAcurácia: {acc:.4f}")
    print(f"F1 macro: {f1_macro:.4f}")
    print(f"F1 weighted: {f1_weighted:.4f}")

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    report_txt = classification_report(y_test, y_pred, zero_division=0)
    print("\n" + report_txt)

    # Salvar relatório e métricas
    with open(os.path.join(REPORT_DIR, "baseline_classification_report.txt"), "w") as f:
        f.write(f"Acurácia: {acc:.4f}\nF1 macro: {f1_macro:.4f}\nF1 weighted: {f1_weighted:.4f}\n\n")
        f.write(report_txt)

    metrics = {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_classes": int(y.nunique()),
    }
    with open(os.path.join(REPORT_DIR, "baseline_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Piores classes (F1 mais baixo) e melhores classes, para análise de erros
    per_class = (
        pd.DataFrame(report).T.drop(index=["accuracy", "macro avg", "weighted avg"])
        .sort_values("f1-score")
    )
    per_class.to_csv(os.path.join(REPORT_DIR, "baseline_per_class_metrics.csv"))
    print("\n=== 8 categorias com pior F1 (candidatas a erro/ambiguidade) ===")
    print(per_class.head(8)[["precision", "recall", "f1-score", "support"]])
    print("\n=== 8 categorias com melhor F1 ===")
    print(per_class.tail(8)[["precision", "recall", "f1-score", "support"]])

    # Salvar modelo e vectorizer
    joblib.dump(clf, os.path.join(MODEL_DIR, "baseline_logreg.joblib"))
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib"))
    print(f"\nModelo salvo em {MODEL_DIR}/baseline_logreg.joblib")

    # Guardar amostra de teste para inspeção/demonstração posterior
    test_sample = pd.DataFrame({"clause": X_test, "true_label": y_test, "pred_label": y_pred})
    test_sample.to_csv(os.path.join(REPORT_DIR, "baseline_test_predictions.csv"), index=False)


if __name__ == "__main__":
    main()
