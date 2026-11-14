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
| `models/best/` | Modelo escolhido (regra de decisão): `mlp_model.pth`, `preprocessor.pkl`, `label_mapping.json`, `metadata.json`. |
| `models/experiments/` | Resultados dos experimentos de arquitetura: `metrics_comparison.csv` (execução única, seções 5-8 do notebook) e `selecao_arquitetura/` (3 arquiteturas × 2 critérios × 5 seeds, `resultados_seeds.csv`, `historicos_seeds.json`, `resultados_opcional.csv`). |
| `reports/` | Documentação final: slides, imagens (`images/curvas_perda.png`, `images/comparacao_seeds.png`), métricas. |
| `tests/` | Testes unitários para validar pipeline. |

### notebooks/ — Etapas de desenvolvimento

| Notebook | Propósito | Saída |
|----------|-----------|-------|
| `01_exploracao_dados.ipynb` | Análise exploratória da SSP-SP (distribuições, colunas, filtros, validade). | Gráficos, filtros validados. |
| `02_preparacao_dados.ipynb` | Consolidação de boletins (múltiplas linhas → 1 boletim) e split estratificado 70/15/15. | `data/processed/train.csv`, `val.csv`, `test.csv`. |
| `03_treinamento_MLP.ipynb` | Treino de 3 arquiteturas MLP (32, 64→32, 128→64→32), comparação por F1-macro em validação, seleção rigorosa com 5 seeds e regra de parcimônia (seção 10), confirmação no teste 2025 (seção 11). | `models/best/` com `mlp_model.pth`, `preprocessor.pkl`, `label_mapping.json`, `metadata.json`; `models/experiments/metrics_comparison.csv`; figuras em `reports/images/`. |
| `dataset_analysis.ipynb` | Análise exploratória inicial (redundante com 01 após refatoração). | Gráficos de distribuição. |

### src/data/ — Preparação de dados

Três módulos principais:

- **loader.py**: Leitura de arquivos Excel da SSP-SP, validação de colunas obrigatórias, filtro por município (S.PAULO).
- **preprocessor.py**: Pipeline scikit-learn para imputação, OneHotEncoding e StandardScaler; ajustado apenas no treino.
- **splitter.py**: Consolidação por boletim de ocorrência (NUM_BO) e separação estratificada 70/15/15.

### Onde ficam os modelos experimentados
Resultados dos experimentos de arquitetura ficam em **`models/experiments/`**:

```
models/experiments/
├── metrics_comparison.csv   # Tabela da célula 8 do notebook (execução única, melhor época, Val Loss, Val Acc, Test Prec/Rec/F1)
└── selecao_arquitetura/
    ├── resultados_seeds.csv     # 30 execuções: 3 arquiteturas × 2 critérios (val_loss, val_f1) × 5 seeds (0-4)
    ├── historicos_seeds.json    # Histórico por época de cada execução (perda e Val F1)
    └── resultados_opcional.csv  # Etapa 8 opcional (parcial): 128→64→32 com weight_decay × dropout × lr, 3 seeds
```

O modelo final selecionado fica em **`models/best/`** (`mlp_model.pth`, `preprocessor.pkl`, `label_mapping.json`, `metadata.json`). Experimentos descartados vão para **`models/archive/`**.

## ▶️ Como executar

1. Crie e ative um ambiente virtual:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate      # Windows
   # source .venv/bin/activate  # Linux/Mac
   ```

2. Instale o pacote em modo desenvolvimento (resolve imports do `src`):
   ```bash
   pip install -e .
   ```
   > Isso lê o `pyproject.toml`, instala dependências de `requirements.txt` e torna `src` importável de qualquer script, notebook ou teste.

   **Alternativa** (sem `pyproject.toml`):
   ```bash
   pip install -r requirements.txt
   ```
   Neste caso, execute scripts da raiz com `python -m Testes_de_Etapas.Etapa1` ou configure `PYTHONPATH`.

3. Execute os notebooks em ordem:
   - `notebooks/01_exploracao_dados.ipynb`
   - `notebooks/02_preparacao_dados.ipynb`
   - `notebooks/03_treinamento_MLP.ipynb`
   - `notebooks/04_avaliacao_final.ipynb`

   > No `03_treinamento_MLP.ipynb`, as seções 5-7 treinam as 3 arquiteturas na GPU (com early stopping, pararam nas épocas 31, 19 e 17). A seção 10 reaproveita `models/experiments/selecao_arquitetura/resultados_seeds.csv` e `historicos_seeds.json` se existirem; caso contrário, treina as 30 execuções (cerca de 1h30 na GPU).

3a. (Opcional) Rodar a seleção de arquitetura fora do notebook. Executar **na raiz do projeto**, com o ambiente ativado e o pacote instalado (`pip install -e .`):
   ```powershell
   # Define UTF-8 na saída do Python (evita erro de codificação do console cp1252 do Windows com '→').
   $env:PYTHONIOENCODING='utf-8'
   # Etapa 5: treina 3 arquiteturas × 2 critérios de early stopping × 5 seeds; grava resultados_seeds.csv e historicos_seeds.json.
   python -m src.models.selecao_arquitetura --etapa seeds
   # Etapas 6-7: lê o CSV, imprime média ± desvio, testes pareados e a regra de decisão; gera as figuras em reports/images/.
   python -m src.models.selecao_arquitetura --etapa relatorio
   # Etapa 8 (opcional, até ~2h): grade de regularização na 128→64→32; grava resultados_opcional.csv.
   python -m src.models.selecao_arquitetura --etapa opcional
   ```
   - Pré-requisitos: arquivos `data/raw/CelularesSubtraidos_2025.xlsx` e GPU CUDA (roda em CPU, porém bem mais lento).
   - Efeito: o CSV é regravado após cada execução; se o script for interrompido, `--etapa seeds` retoma de onde parou (pula execuções com resultado e histórico já salvos).
   - Determinismo: seeds fixas (torch, numpy, random) e cuDNN determinístico; duas rodadas independentes das 10 primeiras execuções deram diferença 0,0 em todas as métricas.

4. Rode a aplicação web (exemplo com Flask/Streamlit):
   ```bash
   python src/webapp/app.py
   ```

## 📊 Entregas obrigatórias

| Entrega | Local |
|---------|-------|
| Repositório GitHub | Este repositório |
| Notebook documentado | `notebooks/` (03_treinamento_MLP.ipynb corrigido: Val Acc da melhor época; seção 10-13 de seleção rigorosa) |
| Relatório técnico (máx. 10 slides) | `reports/slides/` |
| Aplicação web funcional | `src/webapp/` + link no README |
| Modelo selecionado e artefatos | `models/best/` (64→32, val_loss) + `metadata.json` |
| Experimentação completa | `models/experiments/selecao_arquitetura/` (30 seeds, históricos, etapa 8 opcional parcial) |
| Figuras | `reports/images/curvas_perda.png` e `reports/images/comparacao_seeds.png` |
| Tabela de comparação (execução única) | `models/experiments/metrics_comparison.csv` |

## 📦 Dependências

Consulte `requirements.txt`.

## 📝 Fonte dos dados

- **Fonte:** Secretaria da Segurança Pública do Estado de São Paulo (SSP-SP), Portal de Estatísticas — Consultas, base "Celulares subtraídos": <https://www.ssp.sp.gov.br/estatistica/consultas>.
- **Arquivos:** `data/raw/CelularesSubtraidos_2025.xlsx` e `data/raw/CelularesSubtraidos_2026.xlsx`.
- **Data de extração:** *pendente de registro pelo responsável.*
- **Filtros:** município `S.PAULO`; rubricas de roubo (alvo = 1) e furto (alvo = 0).
- **Unidade de análise:** boletim de ocorrência (linhas do mesmo `NUM_BO` consolidadas em uma antes da divisão).

| Recorte | Linhas brutas (furto / roubo) | Boletins consolidados | Uso |
|---|---|---|---|
| 2025 | 183.798 (106.392 / 77.406) | Treino 113.129, validação 24.242, teste 24.243 (70/15/15 estratificado) | Treino, seleção e confirmação |
| 2026 | 126.567 (77.003 / 49.564) | 112.192 | Teste final reservado; **não usado** na Etapa 2 |

- Teste 2025 por classe (boletins): furto 15.291, roubo 8.952.
- Após o pré-processamento (ajustado só no treino), as entradas têm 1.914 colunas.

---

## 🧪 Etapa 2 — Treinamento e seleção da arquitetura MLP

Resultados obtidos na execução completa de `notebooks/03_treinamento_MLP.ipynb` (18 células de código, sem erros, GPU CUDA) em 2026-11-14.

### Configuração comum

| Item | Valor |
|---|---|
| Entrada | 1.914 colunas (one-hot + escala, ajustadas só no treino) |
| Camadas ocultas | ReLU + Dropout 0,2 após cada camada |
| Saída | 1 logit; `BCEWithLogitsLoss`; limiar 0,5 |
| Otimizador | AdamW, lr = 1e-4, weight_decay = 1e-4 |
| Batch | 32 (treino) |
| Épocas | teto de 50; early stopping com patience = 10, min_delta = 1e-4, pesos da melhor época restaurados |
| Métrica de seleção | F1 macro em validação |

### O que foi testado

1. **Execução única (seções 5-8 do notebook)**: 3 arquiteturas, early stopping por `val_loss`, `DataLoader` com `shuffle=True`.

   | Arquitetura | Parâmetros | Parada / melhor época | Val Loss | Val Acc | Val F1 | Test F1 2025 | Test Prec | Test Rec |
   |---|---|---|---|---|---|---|---|---|
   | 32 | 61.313 | 31 / 21 | 0,3824 | 0,8337 | 0,8232 | 0,8248 | 0,7681 | 0,7948 |
   | 64→32 | 124.673 | 19 / 9 | 0,3815 | 0,8340 | 0,8232 | 0,8232 | 0,7705 | 0,7863 |
   | 128→64→32 | 255.489 | 17 / 7 | 0,3799 | 0,8345 | 0,8232 | 0,8237 | 0,7742 | 0,7823 |

   - As 3 arquiteturas empatam em Val F1 até a 4ª casa (0,82318 / 0,82322 / 0,82325). Pelo critério antigo (maior Val F1), a 128→64→32 seria escolhida por 0,00007, uma diferença sem significado.
   - Quanto maior a rede, mais cedo ocorre a melhor época e mais a val_loss sobe depois dela (128→64→32: de 0,3799 na época 7 para 0,3875 na 17), sinal de overfitting mais rápido.

2. **Seleção com várias seeds (seção 10; `src/models/selecao_arquitetura.py`)**: 3 arquiteturas × 2 critérios de early stopping (`val_loss`/min, `val_f1`/max) × 5 seeds (0-4) = 30 execuções determinísticas. Média ± desvio:

   | ES | Arquitetura | Melhor época | Val Loss | Val F1 | Test F1 2025 | Test Prec | Test Rec | Tempo (s) |
   |---|---|---|---|---|---|---|---|---|
   | val_loss | 32 | 27,6 ± 3,9 | 0,3828 ± 0,0004 | 0,8225 ± 0,0007 | 0,8231 ± 0,0004 | 0,7679 ± 0,0018 | 0,7898 ± 0,0033 | 231 ± 25 |
   | val_loss | 64→32 | 10,8 ± 1,3 | 0,3809 ± 0,0002 | 0,8235 ± 0,0005 | 0,8232 ± 0,0008 | 0,7704 ± 0,0034 | 0,7865 ± 0,0062 | 163 ± 11 |
   | val_loss | 128→64→32 | 7,0 ± 1,2 | 0,3803 ± 0,0003 | 0,8235 ± 0,0007 | 0,8232 ± 0,0011 | 0,7698 ± 0,0029 | 0,7873 ± 0,0057 | 163 ± 13 |
   | val_f1 | 32 | 28,4 ± 12,3 | 0,3833 ± 0,0005 | 0,8236 ± 0,0003 | 0,8236 ± 0,0007 | 0,7648 ± 0,0034 | 0,7960 ± 0,0050 | 232 ± 68 |
   | val_f1 | 64→32 | 7,6 ± 3,0 | 0,3826 ± 0,0012 | 0,8238 ± 0,0005 | 0,8239 ± 0,0007 | 0,7649 ± 0,0060 | 0,7971 ± 0,0099 | 137 ± 23 |
   | val_f1 | 128→64→32 | 6,6 ± 1,5 | 0,3804 ± 0,0004 | 0,8240 ± 0,0004 | 0,8239 ± 0,0006 | 0,7691 ± 0,0052 | 0,7907 ± 0,0096 | 159 ± 15 |

3. **Testes pareados por seed sobre Val F1 (n = 5)**:

   | ES | Comparação | Dif. média | Vence em | p (t pareado) | p (Wilcoxon) |
   |---|---|---|---|---|---|
   | val_loss | 64→32 − 32 | +0,0010 | 5/5 | 0,023 | 0,0625 |
   | val_loss | 128→64→32 − 32 | +0,0010 | 4/5 | 0,045 | 0,125 |
   | val_loss | 128→64→32 − 64→32 | −0,0000 | 3/5 | 0,950 | 1,000 |
   | val_f1 | 64→32 − 32 | +0,0003 | 3/5 | 0,470 | 0,438 |
   | val_f1 | 128→64→32 − 32 | +0,0004 | 4/5 | 0,240 | 0,188 |
   | val_f1 | 128→64→32 − 64→32 | +0,0002 | 2/5 | 0,643 | 0,813 |

   - Com n = 5, o menor p possível no Wilcoxon bilateral é 0,0625; o poder estatístico é baixo e um p alto não prova equivalência.
   - Só a vantagem das redes de 2 e 3 camadas sobre a 32, com ES por val_loss, aparece no t pareado (≈ 0,001 de F1). Entre 64→32 e 128→64→32 não há diferença em nenhum critério.

4. **Etapa 8 opcional (parcial)**: 128→64→32 com `weight_decay ∈ {1e-4, 1e-3}` × `dropout ∈ {0,3; 0,5}` × `lr ∈ {1e-4, 5e-5}`, 3 seeds, ES por val_loss. Foram concluídas 9 das 24 execuções; a execução foi interrompida a pedido do responsável.

   | weight_decay | Dropout | lr | Melhor época | Val F1 | Test F1 2025 |
   |---|---|---|---|---|---|
   | 1e-4 | 0,3 | 1e-4 | 8,0 ± 1,0 | 0,8237 ± 0,0005 | 0,8237 ± 0,0012 |
   | 1e-4 | 0,3 | 5e-5 | 13,3 ± 0,6 | 0,8231 ± 0,0007 | 0,8228 ± 0,0014 |
   | 1e-4 | 0,5 | 1e-4 | 8,7 ± 0,6 | 0,8238 ± 0,0002 | 0,8241 ± 0,0002 |

   - Nenhuma combinação concluída supera a 128→64→32 base (0,8235 ± 0,0007) nem a 64→32 (0,8235 ± 0,0005) além da variação entre seeds. Mais regularização não justificou a rede maior.

5. **Determinismo**: as 10 primeiras execuções da seção 10 foram feitas duas vezes, de forma independente; a diferença foi 0,0 em melhor época, val_loss, val_F1 e test_F1.

### Regra de decisão e modelo escolhido

**Regra**: escolher a arquitetura de maior Val F1 médio (5 seeds). Se uma arquitetura mais simples ficar a até 1 desvio padrão (o maior dos dois), escolhe-se a mais simples. O teste 2025 é consultado só depois, para confirmar; o teste 2026 não é usado.

| ES | Resultado da regra | Justificativa |
|---|---|---|
| val_loss | **64→32** | 64→32 e 128→64→32 empatam em 0,8235; a 32 (0,8225) fica a 0,0010, fora de 1 desvio (0,0007). A regra fica com a 64→32, a mais simples entre as empatadas. |
| val_f1 | 32 | 128→64→32 lidera com 0,8240, mas a 32 fica a 0,0004, dentro de 1 desvio (0,0004). |

**Modelo adotado: 64→32 com early stopping por val_loss** (`models/best/`). A decisão considerou todas as variáveis medidas:

- **Desempenho (Val F1)**: empata com a 128→64→32 (0,8235 nas duas; p = 0,95) e supera a 32 em 0,0010, a única diferença com indício estatístico (5/5 seeds, t pareado p = 0,023).
- **Val Loss**: 0,3809 ± 0,0002, o menor desvio entre as arquiteturas; a 128→64→32 tem média ligeiramente menor (0,3803), mas sem ganho em F1.
- **Complexidade**: 124.673 parâmetros, metade da 128→64→32 (255.489), com menor risco de overfitting e inferência mais leve para a aplicação web.
- **Estabilidade do treino**: melhor época 10,8 ± 1,3 e val_loss subindo menos depois do mínimo do que na 128→64→32; a 32 precisa de ~28 épocas e varia mais (± 12 épocas com ES por val_f1).
- **Tempo de treino**: 163 ± 11 s, igual ao da 128→64→32 e menor que o da 32 (231 s), que converge mais devagar.
- **Precisão × recall (roubo)**: diferenças entre arquiteturas ≤ 0,006, menores que a variação entre seeds; nenhuma arquitetura traz ganho claro em uma classe.
- **Critério de early stopping**: o `val_loss` foi mantido como padrão. O `val_f1` sobe o F1 médio em ~0,0003-0,001, mas deixa a melhor época mais instável (32: ± 12,3; 64→32: ± 3,0) e aumenta o desvio da val_loss; a perda é um sinal mais suave para parar.
- **Por que não a 32 (resultado da regra com val_f1)**: com o critério padrão (val_loss), ela fica 0,0010 abaixo de forma consistente; com val_f1, as três ficam dentro do ruído. A 64→32 é a escolha que não perde em nenhum dos dois critérios.

**Confirmação no teste 2025** (modelo salvo, 64→32, melhor época 9 da execução única):

| Classe | Precisão | Recall | F1 | Boletins |
|---|---|---|---|---|
| Furto (0) | 0,87 | 0,86 | 0,87 | 15.291 |
| Roubo (1) | 0,77 | 0,79 | 0,78 | 8.952 |
| **Macro** | 0,82 | 0,82 | **0,8232** | 24.243 |

Acurácia 0,8346. O resultado coincide com a média das 5 seeds (test F1 0,8232 ± 0,0008), o que indica que o modelo salvo é representativo.

### Figuras

- `reports/images/curvas_perda.png`: perda de treino/validação e Val F1 por época (seed 0, ES por val_loss), com a melhor época tracejada e o mínimo da val_loss marcado.
- `reports/images/comparacao_seeds.png`: boxplot e pontos por seed de Val F1 (seleção) e Test F1 2025 (confirmação), por arquitetura e critério.

---

## Histórico de decisões

| Data | Decisão | Estado | Motivo | Consequência |
|------|---------|--------|--------|--------------|
| 2026-11-12 | Usar `pyproject.toml` + `pip install -e .` para instalar o pacote | **DECIDIDO** | Resolve `ModuleNotFoundError: No module named 'src'` ao executar scripts fora da raiz; padrão Python moderno para pacotes editáveis. | `src` importável em qualquer contexto (scripts, notebooks, testes, webapp); dependências declaradas em um só lugar. |
| 2026-11-10 | Usar `openpyxl` para leitura de .xlsx | **DECIDIDO** | Arquivos SSP-SP são `.xlsx`; pandas requer motor adicional. | Adicionado a `requirements.txt`. |
| 2026-11-10 | Escolher 4 features principais para a MLP | **DECIDIDO** | Atributos disponíveis e pertinentes ao problema: `DESCR_TIPOLOCAL`, `DESCR_SUBTIPOLOCAL`, `BAIRRO`, `DESCR_PERIODO`. | Derivados temporais (`dia_semana`, `mes`, `hora`) incluídos automaticamente. |
| 2026-11-10 | Excluir `DESCR_CONDUTA`, `CIRCUNSTANCIA`, `MARCA_OBJETO` | **DECIDIDO** | Não correlacionam com roubo vs furto; aumentam dimensionalidade sem benefício preditivo. | Redução de features, evita overfit. |
| 2026-11-10 | Target binário: `RUBRICA` → 0 (furto), 1 (roubo) | **DECIDIDO** | Conforme enunciado da Global Solution. | Registro de `target` em DataFrame. |
| 2026-11-12 | Framework de deep learning: PyTorch (vs. TensorFlow/Keras) | **DECIDIDO** | PyTorch oferece maior controle sobre arquiteturas, loops de treino e dropout customizável; comunidade ativa em pesquisa. | Implementação de MLP com `nn.Module`, `BCEWithLogitsLoss`, Adam; artefatos em `.pth`. |
| 2026-11-12 | Métrica de seleção: F1-score macro (validação) | **DECIDIDO** | Dataset levemente desbalanceado (mais furtos que roubos); F1 macro dá peso igual a ambas as classes, essencial para segurança pública. | Arquitetura com melhor F1-macro em validação é eleita para teste final (2026). |
| 2026-11-12 | Regularização: Dropout 0.2 em cada camada oculta | **DECIDIDO** | Primeiros testes mostraram overfitting em todas as arquiteturas; Dropout reduz co-adaptação de neurônios. | Adicionado a cada camada oculta antes da ativação; early stopping em desenvolvimento. |
| 2026-11-12 | Função de perda: BCEWithLogitsLoss (sigmoid integrada) | **DECIDIDO** | Numericamente mais estável que BCE + Sigmoid separadas; recomendado para classificação binária em PyTorch. | Saída da rede é logit bruto; sigmoide aplicada internamente na perda. |
| 2026-11-12 | Dependência: joblib para serializar preprocessor | **DECIDIDO** | scikit-learn's `MinMaxScaler`, `OneHotEncoder`, etc. não são serializáveis com `pickle` vanilla em todos os ambientes; `joblib` é padrão. | `preprocessor.pkl` carregado em inferência para aplicar mesmas transformações do treino. |
| 2026-11-14 | Corrigir tabela da célula 8 do notebook `03_treinamento_MLP.ipynb` | **DECIDIDO** | `Val Acc` vinha de `val_accs_k[-1]` (última época) enquanto `Val F1` usava a melhor época; comparação entre métricas de épocas diferentes era inválida. | `Val Acc` agora usa `val_acc_final_*` (mesma avaliação da melhor época, pesos restaurados). Acrescentadas colunas "Melhor época" e "Val Loss". |
| 2026-11-14 | Criar `src/models/selecao_arquitetura.py` (724 linhas) | **DECIDIDO** | Substitui a decisão baseada em execução única (1 seed) por seleção com 5 seeds, t pareado e Wilcoxon. | Early stopping configurável por `monitor` (`val_loss`/min, `val_f1`/max) e `mode`; treino GPU determinístico; salvamento incremental de CSV + JSON com retomada; tabelas média ± desvio; testes pareados (n=5); figuras; regra de decisão por parcimônia; etapa 8 opcional (grade de regularização). Substitui a coluna "Val F1" da célula 8 como critério de seleção final. |
| 2026-11-14 | Adicionar seções 10-13 ao notebook | **DECIDIDO** | A seleção de uma execução única (célula 8) era ruidosa; as diferenças entre arquiteturas (~0,002) são menores que a variação entre seeds. | Seção 10: 30 execuções (seeds), média ± desvio, testes pareados, regra de decisão. Seção 11: matriz de confusão do modelo escolhido pela regra (confirmação no teste 2025). Seção 12: salva `models/best/` com metadados. Seção 13: justificativa. Células 10-11 originais (matriz de confusão/salvar modelo) reescritas para usar a escolha da regra, não mais a 128→64→32 do critério antigo. |
| 2026-11-14 | Procedimento anterior: escolher pelo maior Val F1 de uma execução única (`np.argmax` na célula 8) | **SUBSTITUÍDO** | Na execução única as 3 arquiteturas ficaram em 0,82318 / 0,82322 / 0,82325; o critério escolheria a 128→64→32 por 0,00007, diferença menor que a variação entre seeds (dp ≈ 0,0005). | Sucessor: regra de decisão com 5 seeds e parcimônia (`regra_decisao` em `src/models/selecao_arquitetura.py`, seções 10-11 do notebook). A célula 8 continua exibindo a tabela, agora só como descrição. A decisão de 2026-11-12 "Métrica de seleção: F1 macro (validação)" continua válida; muda apenas o modo de aplicá-la (média de 5 seeds em vez de 1 execução). |
| 2026-11-14 | Critério de early stopping: manter `val_loss` como padrão; `val_f1` testado como alternativa | **DECIDIDO** | Com `val_f1`, o F1 médio sobe só ~0,0003-0,001 (dentro do ruído), mas a melhor época fica instável (32: 28,4 ± 12,3) e a val_loss piora. | `monitor='val_loss'`, `mode='min'`, patience 10, min_delta 1e-4. `val_f1` permanece disponível no script para comparação. |
| 2026-11-14 | Etapa 8 opcional: regularizar a 128→64→32 (weight_decay × dropout × lr, 3 seeds) | **INTERROMPIDO** | Executada para verificar se a rede maior se justificaria com mais regularização; interrompida a pedido do responsável após 9 de 24 execuções. | Melhor combinação concluída (wd 1e-4, dropout 0,5, lr 1e-4): Val F1 0,8238 ± 0,0002, sem ganho além do ruído. Resultados parciais em `resultados_opcional.csv`; as 15 combinações restantes (wd 1e-4/dropout 0,5/lr 5e-5 e todas com wd 1e-3) podem ser rodadas com `--etapa opcional`. |
| 2026-11-14 | Escolher modelo final: 64→32 com early stopping por `val_loss` | **DECIDIDO** | 64→32 e 128→64→32 empatam em Val F1 (0,8235; p=0,95 t pareado); a 32 fica 0,0010 abaixo. 64→32 tem metade dos parâmetros, Val Loss mais estável, menor risco de overfitting (melhor época mais tardia) e não perde para a 128→64→32 em nenhum critério. O teste 2025 confirma (F1 0,8232). A etapa 8 opcional (128→64→32 com regularização; 9/24 combinações) não mostrou justificativa para a rede maior. | Modelo salvo em `models/best/mlp_model.pth` + `metadata.json` (arquitetura, hiperparâmetros, regra, métricas de validação e confirmação no teste 2025). Teste 2026 continua reservado e fora da seleção. |
