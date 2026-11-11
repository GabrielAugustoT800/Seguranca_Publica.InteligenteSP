# Segurança Pública Inteligente SP — Global Solution

Classificação de ocorrências da cidade de São Paulo como **roubo** ou **furto** usando uma rede neural MLP (Multilayer Perceptron).

## 📋 Descrição

Desafio prático de 4 horas (disciplina: Aprendizado Profundo com Redes Neurais Convolucionais). O objetivo é construir, treinar, avaliar e comparar pelo menos 3 arquiteturas de MLP, selecionar o melhor modelo e integrá-lo a uma aplicação web simples que permita classificar novas ocorrências por meio de um formulário.

> **Importante:** o modelo prevê o tipo da ocorrência (roubo ou furto) a partir das características de um registro **já ocorrido**. Ele não calcula a chance de um crime acontecer em determinado local ou horário.

## 🗂️ Estrutura do projeto

| Pasta | Propósito |
|-------|-----------|
| `data/` | Dados brutos (`raw/`) e processados (`processed/`); não versionados via .gitignore. |
| `notebooks/` | Exploração e desenvolvimento iterativo; notebooks documentados. |
| `src/` | Código fonte modular: carregamento, pré-processamento, modelos, inferência, webapp. |
| `models/` | Artefatos treinados (`best/`, `experiments/`, `archive/`); versionados via instruções. |
| `reports/` | Documentação final: slides, imagens, métricas. |
| `tests/` | Testes unitários para validar pipeline. |

### src/data/ — Preparação de dados

Três módulos principais:

- **loader.py**: Leitura de arquivos Excel da SSP-SP, validação de colunas obrigatórias, filtro por município (S.PAULO).
- **preprocessor.py**: Pipeline scikit-learn para imputação, OneHotEncoding e StandardScaler; ajustado apenas no treino.
- **splitter.py**: Consolidação por boletim de ocorrência (NUM_BO) e separação estratificada 70/15/15.

### Onde ficam os modelos experimentados
Todos os modelos treinados durante a comparação de arquiteturas ficam em **`models/experiments/`**, organizados por experimento:

```
models/experiments/
├── mlp_32_relu_sigmoid/
│   └── model_final.pth
├── mlp_64_32_relu_adam/
│   └── model_final.pth
├── mlp_128_64_32_relu/
│   └── model_final.pth
└── metrics_comparison.csv   # Resumo de acurácia, precisão, recall e F1 de cada experimento
```

O modelo final selecionado fica em **`models/best/`**. Experimentos descartados vão para **`models/archive/`**.

## ▶️ Como executar

1. Crie e ative um ambiente virtual:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate      # Windows
   # source .venv/bin/activate  # Linux/Mac
   ```

2. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

3. Execute os notebooks em ordem:
   - `notebooks/01_exploracao_dados.ipynb`
   - `notebooks/02_preparacao_dados.ipynb`
   - `notebooks/03_treinamento_mlp.ipynb`
   - `notebooks/04_avaliacao_final.ipynb`

4. Rode a aplicação web (exemplo com Flask/Streamlit):
   ```bash
   python src/webapp/app.py
   ```

## 📊 Entregas obrigatórias

| Entrega | Local |
|---------|-------|
| Repositório GitHub | Este repositório |
| Notebook documentado | `notebooks/` |
| Relatório técnico (máx. 10 slides) | `reports/slides/` |
| Aplicação web funcional | `src/webapp/` + link no README |

## 📦 Dependências

Consulte `requirements.txt`.

## 📝 Fonte dos dados

Registrar no README o link da fonte pública escolhida (ex: SSP-SP), data de acesso e período dos registros utilizados.

---

## Histórico de decisões

| Data | Decisão | Estado | Motivo | Consequência |
|------|---------|--------|--------|--------------|
| 2026-11-10 | Usar `openpyxl` para leitura de .xlsx | **DECIDIDO** | Arquivos SSP-SP são `.xlsx`; pandas requer motor adicional. | Adicionado a `requirements.txt`. |
| 2026-11-10 | Escolher 4 features principais para a MLP | **DECIDIDO** | Atributos disponíveis e pertinentes ao problema: `DESCR_TIPOLOCAL`, `DESCR_SUBTIPOLOCAL`, `BAIRRO`, `DESCR_PERIODO`. | Derivados temporais (`dia_semana`, `mes`, `hora`) incluídos automaticamente. |
| 2026-11-10 | Excluir `DESCR_CONDUTA`, `CIRCUNSTANCIA`, `MARCA_OBJETO` | **DECIDIDO** | Não correlacionam com roubo vs furto; aumentam dimensionalidade sem benefício preditivo. | Redução de features, evita overfit. |
| 2026-11-10 | Target binário: `RUBRICA` → 0 (furto), 1 (roubo) | **DECIDIDO** | Conforme enunciado da Global Solution. | Registro de `target` em DataFrame. |
