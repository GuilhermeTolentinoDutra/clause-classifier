"""Fase 1 - Download do dataset CUAD (versão de classificação de cláusulas).

Fonte: dvgodoy/CUAD_v1_Contract_Understanding_clause_classification (Hugging Face)
Origem original: The Atticus Project - CUAD v1 (https://www.atticusprojectai.org/cuad)
13.155 cláusulas rotuladas extraídas de 509 contratos comerciais, em 41 categorias.
"""
from datasets import load_dataset
import pandas as pd
import os

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
os.makedirs(OUT_DIR, exist_ok=True)

def main():
    ds = load_dataset("dvgodoy/CUAD_v1_Contract_Understanding_clause_classification")
    df = ds["train"].to_pandas()
    out_path = os.path.join(OUT_DIR, "cuad_clauses.csv")
    df.to_csv(out_path, index=False)
    print(f"Salvo {len(df)} cláusulas em {out_path}")
    print(f"Número de categorias: {df['label'].nunique()}")

if __name__ == "__main__":
    main()
