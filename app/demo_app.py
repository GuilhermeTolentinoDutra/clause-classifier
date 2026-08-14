"""Fase 1 - Demo interativa (Streamlit) do classificador de cláusulas contratuais.

MVP "walking skeleton": cola-se um contrato (ou uma cláusula avulsa), o texto é
segmentado em blocos e cada bloco é classificado pelo baseline TF-IDF + Regressão
Logística treinado em src/train_baseline.py, sobre o dataset CUAD (41 categorias).

Este é o Nível 1 do roadmap. Nas próximas fases, o modelo por trás desta mesma
interface será substituído por um Legal-BERT fine-tuned (Fase 2), depois comparado
a um LLM (Fase 3) e enriquecido com explicabilidade via SHAP (Fase 4).
"""
import os
import re
import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
MODEL_PATH = os.path.join(BASE_DIR, "data", "processed", "baseline_logreg.joblib")
VEC_PATH = os.path.join(BASE_DIR, "data", "processed", "tfidf_vectorizer.joblib")

EXAMPLE_CLAUSE = (
    "This Agreement shall remain in full force and effect for an initial term of "
    "three (3) years from the Effective Date, and shall automatically renew for "
    "successive one (1) year periods unless either party provides written notice "
    "of non-renewal at least sixty (60) days prior to the end of the then-current term."
)

EXAMPLE_CONTRACT = (
    "This Agreement shall remain in full force and effect for an initial term of three (3) years "
    "from the Effective Date, and shall automatically renew for successive one (1) year periods "
    "unless either party provides written notice of non-renewal at least sixty (60) days prior to "
    "the end of the then-current term.\n\n"
    "Each party agrees to keep confidential all non-public information disclosed by the other party "
    "in connection with this Agreement, and shall not disclose such information to any third party "
    "without prior written consent.\n\n"
    "In no event shall either party's aggregate liability arising out of or related to this Agreement "
    "exceed the total fees paid by Client to Vendor during the twelve (12) months preceding the claim.\n\n"
    "This Agreement shall be governed by and construed in accordance with the laws of the State of "
    "Delaware, without regard to its conflict of laws principles.\n\n"
    "Neither party may assign this Agreement, in whole or in part, without the prior written consent "
    "of the other party, except in connection with a merger, acquisition, or sale of substantially all "
    "of its assets."
)


@st.cache_resource
def load_model():
    clf = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VEC_PATH)
    return clf, vectorizer


def segment_text(raw_text: str):
    """Segmentação simples de um contrato em blocos candidatos a cláusula.

    Heurística do MVP: separa por linhas em branco e, dentro de blocos longos,
    por sentenças. Isso é suficiente para o walking skeleton; um segmentador mais
    robusto (baseado em numeração de cláusulas, seções, etc.) é um item de
    melhoria natural para a Fase 2+.
    """
    blocks = [b.strip() for b in re.split(r"\n\s*\n", raw_text) if b.strip()]
    segments = []
    for block in blocks:
        if len(block.split()) > 120:
            sentences = re.split(r"(?<=[.;])\s+", block)
            segments.extend([s.strip() for s in sentences if len(s.strip()) > 15])
        else:
            segments.append(block)
    return segments


def classify(segments, clf, vectorizer, top_k=3):
    X = vectorizer.transform(segments)
    probs = clf.predict_proba(X)
    classes = clf.classes_
    results = []
    for i, seg in enumerate(segments):
        order = np.argsort(probs[i])[::-1][:top_k]
        top_preds = [(classes[j], float(probs[i][j])) for j in order]
        results.append({"clause": seg, "top_predictions": top_preds})
    return results


def main():
    st.set_page_config(page_title="Classificador de Cláusulas Contratuais", layout="wide")
    st.title("Classificador de Cláusulas Contratuais — MVP (Fase 1)")
    st.caption(
        "Projeto de portfólio Direito + Machine Learning · Modelo atual: TF-IDF + Regressão Logística "
        "treinado no dataset CUAD (41 categorias de cláusulas, 509 contratos comerciais). "
        "Este é um artefato educacional/portfólio, não uma ferramenta de aconselhamento jurídico."
    )

    with st.sidebar:
        st.header("Sobre este MVP")
        st.markdown(
            "- **Nível técnico atual:** 1/4 (baseline interpretável)\n"
            "- **Dataset:** CUAD v1 (The Atticus Project)\n"
            "- **Próximos passos:** fine-tuning de Legal-BERT, benchmark com LLMs, "
            "explicabilidade (SHAP) e extensão para português.\n"
        )
        top_k = st.slider("Quantidade de categorias sugeridas por trecho", 1, 5, 3)

    try:
        clf, vectorizer = load_model()
    except FileNotFoundError:
        st.error(
            "Modelo não encontrado. Rode `python src/train_baseline.py` antes de iniciar a demo."
        )
        st.stop()

    mode = st.radio(
        "O que você quer classificar?",
        ["Uma cláusula avulsa", "Um contrato inteiro (várias cláusulas)"],
        horizontal=True,
    )

    if mode == "Uma cláusula avulsa":
        text = st.text_area("Cole o texto da cláusula (em inglês):", value=EXAMPLE_CLAUSE, height=140)
        if st.button("Classificar cláusula", type="primary"):
            if not text.strip():
                st.warning("Cole um texto antes de classificar.")
            else:
                result = classify([text], clf, vectorizer, top_k=top_k)[0]
                st.subheader("Resultado")
                for label, prob in result["top_predictions"]:
                    st.progress(min(prob, 1.0), text=f"{label} — {prob*100:.1f}%")
    else:
        text = st.text_area(
            "Cole o texto do contrato (separe cláusulas por linha em branco):",
            value=EXAMPLE_CONTRACT,
            height=260,
        )
        if st.button("Classificar contrato", type="primary"):
            if not text.strip():
                st.warning("Cole um texto antes de classificar.")
            else:
                segments = segment_text(text)
                if not segments:
                    st.warning("Não foi possível identificar blocos de texto.")
                else:
                    results = classify(segments, clf, vectorizer, top_k=top_k)
                    st.subheader(f"{len(results)} trecho(s) identificado(s)")
                    for i, r in enumerate(results, 1):
                        top_label, top_prob = r["top_predictions"][0]
                        with st.expander(f"Trecho {i}: {top_label} ({top_prob*100:.1f}%)"):
                            st.write(r["clause"])
                            st.markdown("**Top categorias sugeridas:**")
                            for label, prob in r["top_predictions"]:
                                st.write(f"- {label}: {prob*100:.1f}%")

    st.divider()
    st.caption(
        "Dataset CUAD: The Atticus Project (https://www.atticusprojectai.org/cuad/). "
        "Este projeto não constitui aconselhamento jurídico e não deve ser usado em produção."
    )


if __name__ == "__main__":
    main()
