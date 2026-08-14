# Classificador de Cláusulas Contratuais — Fase 1 (MVP / Walking Skeleton)

Projeto de portfólio na interseção **Direito + Machine Learning**. Classifica automaticamente cláusulas de contratos comerciais em categorias jurídicas (ex.: *Governing Law, Non-Compete, Cap On Liability, IP Ownership Assignment*).

> Este é um artefato educacional/portfólio, **não** uma ferramenta de aconselhamento jurídico e não deve ser usado em produção sem validação adicional.

## Status atual: Fase 1 concluída (Nível técnico 1/4)

Este é o "walking skeleton" do roadmap: um pipeline completo e funcional, do dado bruto até uma demo interativa, usando o modelo mais simples possível (baseline interpretável). As próximas fases substituem o modelo por versões mais sofisticadas sem alterar a experiência da demo.

| Fase | Descrição | Status |
|---|---|---|
| 1 | Walking skeleton: dataset CUAD + baseline TF-IDF/LogReg + demo | ✅ Concluída |
| 2 | Fine-tuning de Legal-BERT (EN) | Próxima |
| 3 | Benchmark comparativo com LLMs (zero-shot) | Planejada |
| 4 | Explicabilidade (SHAP/LIME) | Planejada |
| 5 | Extensão para português (Legal-BERTimbau + LGPD) | Planejada |
| 6-8 | Polimento, documentação e publicação | Planejadas |

## Dataset

**CUAD (Contract Understanding Atticus Dataset) v1** — 13.155 cláusulas rotuladas, extraídas de 509 contratos comerciais reais, anotadas por advogados sob supervisão do [The Atticus Project](https://www.atticusprojectai.org/cuad/), cobrindo 41 categorias de cláusulas relevantes para revisão contratual (M&A, licenciamento, fornecimento, etc.).

Usamos a versão já formatada para classificação (texto → rótulo), disponível em [`dvgodoy/CUAD_v1_Contract_Understanding_clause_classification`](https://huggingface.co/datasets/dvgodoy/CUAD_v1_Contract_Understanding_clause_classification) no Hugging Face.

### Principais achados da análise exploratória (`reports/`)
- 509 contratos únicos, 41 categorias, cláusulas com em média 43 palavras (mediana 33).
- **Forte desbalanceamento de classes**: a categoria mais frequente (*Parties*, 2.560 exemplos) tem quase **100x** mais exemplos que a menos frequente (*Price Restrictions*, 26 exemplos). Isso foi tratado no baseline com `class_weight="balanced"`, mas seguirá sendo um ponto de atenção nas próximas fases (ex.: oversampling, focal loss, ou agrupamento de categorias raras).
- Gráficos completos em `reports/class_distribution.png` e `reports/clause_length_distribution.png`.

## Modelo baseline (Nível técnico 1)

**TF-IDF (unigrama + bigrama, 20k features) + Regressão Logística** (`class_weight="balanced"`), split estratificado 80/20.

| Métrica | Valor |
|---|---|
| Acurácia | 74,7% |
| F1 macro | 0,63 |
| F1 weighted | 0,75 |

Interpretação: a acurácia geral é sólida para um baseline simples, mas o F1 macro mais baixo confirma o impacto do desbalanceamento — categorias raras (*Affiliate License-Licensor*, *No-Solicit Of Customers*, com <10 exemplos de teste) têm desempenho fraco, enquanto categorias bem representadas e "lexicamente distintas" (*Governing Law*, *Parties*, *Insurance*) alcançam F1 acima de 0,95. Essa análise por classe (`reports/baseline_per_class_metrics.csv`) é o ponto de partida direto para a Fase 2: o fine-tuning de um encoder pré-treinado (Legal-BERT) deve generalizar melhor justamente nas classes raras, pois não depende só de sobreposição literal de vocabulário.

## Demo interativa

App em Streamlit (`app/demo_app.py`) com dois modos:
1. **Cláusula avulsa** — cola-se um trecho, o modelo retorna as top-N categorias mais prováveis com a confiança de cada uma.
2. **Contrato inteiro** — cola-se um texto maior, o app segmenta em blocos (heurística por parágrafo/sentença) e classifica cada um.

### Como rodar localmente
```bash
pip install -r requirements.txt
python src/download_data.py     # baixa o dataset CUAD
python src/eda.py               # gera a análise exploratória em reports/
python src/train_baseline.py    # treina o baseline e salva o modelo em data/processed/
streamlit run app/demo_app.py   # abre a demo em http://localhost:8501
```

## Estrutura do projeto
```
clause-classifier/
├── src/
│   ├── download_data.py   # baixa o CUAD do Hugging Face
│   ├── eda.py              # análise exploratória
│   └── train_baseline.py   # treino e avaliação do baseline
├── app/
│   └── demo_app.py         # demo Streamlit
├── data/
│   ├── raw/                # dataset bruto (CSV)
│   └── processed/          # modelo e vetorizador treinados (.joblib)
├── reports/                 # métricas, gráficos e screenshots
└── requirements.txt
```

## Próximos passos (Fase 2)
- Fine-tuning de `nlpaueb/legal-bert-base-uncased` (ou modelo equivalente) no mesmo split de treino/teste, para comparação direta com o baseline.
- Registrar experimentos (ex.: MLflow) para comparar hiperparâmetros.
- Repetir a análise por classe para verificar se as categorias raras melhoram.

## Fontes
- CUAD dataset: [The Atticus Project](https://www.atticusprojectai.org/cuad/), [Hugging Face](https://huggingface.co/datasets/theatticusproject/cuad)
- Versão de classificação usada: [dvgodoy/CUAD_v1_Contract_Understanding_clause_classification](https://huggingface.co/datasets/dvgodoy/CUAD_v1_Contract_Understanding_clause_classification)
