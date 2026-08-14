"""Fase 1 - Análise exploratória do dataset CUAD (classificação de cláusulas)."""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "cuad_clauses.csv")
REPORT_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")
os.makedirs(REPORT_DIR, exist_ok=True)

def main():
    df = pd.read_csv(DATA_PATH)
    df["n_words"] = df["clause"].astype(str).str.split().str.len()
    df["n_chars"] = df["clause"].astype(str).str.len()

    print("=== Visão geral ===")
    print(f"Total de cláusulas: {len(df)}")
    print(f"Número de contratos únicos: {df['file_name'].nunique()}")
    print(f"Número de categorias: {df['label'].nunique()}")
    print(f"Tamanho médio (palavras): {df['n_words'].mean():.1f} | mediana: {df['n_words'].median():.0f}")
    print(f"Tamanho médio (caracteres): {df['n_chars'].mean():.1f}")

    counts = df["label"].value_counts()
    print("\n=== Distribuição de classes (top 10 maiores) ===")
    print(counts.head(10))
    print("\n=== Distribuição de classes (10 menores) ===")
    print(counts.tail(10))
    print(f"\nRazão desbalanceamento (maior/menor classe): {counts.max()/counts.min():.1f}x")

    counts.to_csv(os.path.join(REPORT_DIR, "class_distribution.csv"))

    # Gráfico de distribuição de classes
    fig, ax = plt.subplots(figsize=(10, 12))
    counts.sort_values().plot(kind="barh", ax=ax, color="#2b6cb0")
    ax.set_xlabel("Número de cláusulas")
    ax.set_title("Distribuição de cláusulas por categoria (CUAD, 41 classes)")
    plt.tight_layout()
    fig.savefig(os.path.join(REPORT_DIR, "class_distribution.png"), dpi=140)
    print(f"\nGráfico salvo em {REPORT_DIR}/class_distribution.png")

    # Distribuição de tamanho de texto
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    df["n_words"].clip(upper=200).hist(bins=50, ax=ax2, color="#2b6cb0")
    ax2.set_xlabel("Número de palavras por cláusula (truncado em 200)")
    ax2.set_ylabel("Frequência")
    ax2.set_title("Distribuição do tamanho das cláusulas")
    plt.tight_layout()
    fig2.savefig(os.path.join(REPORT_DIR, "clause_length_distribution.png"), dpi=140)
    print(f"Gráfico salvo em {REPORT_DIR}/clause_length_distribution.png")

if __name__ == "__main__":
    main()
