"""Seleção rigorosa de arquitetura MLP (classificação binária roubo × furto).

Funções novas; o código original do notebook 03_treinamento_MLP.ipynb não é alterado.
Divisão dos dados e pré-processamento são reutilizados sem mudança (src.data.*).
O teste de 2026 fica fora desta análise. O teste de 2025 só confirma a escolha.

Execução (na raiz do projeto):
    python -m src.models.selecao_arquitetura --etapa seeds      # etapas 5, 6 e 7
    python -m src.models.selecao_arquitetura --etapa opcional   # etapa 8
"""

# Importa o módulo do sistema operacional para definir variáveis de ambiente.
import os
# Configura o workspace do cuBLAS para permitir algoritmos determinísticos na GPU (precisa vir antes do uso de CUDA).
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
# Importa o gerador aleatório padrão do Python para fixar sua semente.
import random
# Importa o módulo de tempo para medir a duração de cada treino.
import time
# Importa o módulo json para salvar os históricos por época em arquivo.
import json
# Importa argparse para ler o parâmetro --etapa da linha de comando.
import argparse

# Importa numpy para operações numéricas.
import numpy as np
# Importa pandas para montar e salvar as tabelas de resultados.
import pandas as pd
# Importa PyTorch para construir e treinar as redes.
import torch
# Importa o módulo de camadas do PyTorch.
import torch.nn as nn
# Importa o otimizador AdamW (o mesmo do notebook original).
from torch.optim import AdamW
# Importa as métricas de classificação do scikit-learn.
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# Importa os testes estatísticos do scipy (t pareado e Wilcoxon).
from scipy import stats
# Importa matplotlib com backend sem janela para salvar figuras em arquivo.
import matplotlib
# Seleciona o backend 'Agg', que grava PNG sem precisar de interface gráfica.
matplotlib.use('Agg')
# Importa a interface de gráficos do matplotlib.
import matplotlib.pyplot as plt

# Importa a função que lê os microdados brutos de um ano e define o alvo.
from src.data.loader import carregar_e_preparar
# Importa a função que consolida boletins e divide 2025 em 70/15/15 (sem alteração).
from src.data.splitter import preparar_split_2025
# Importa a função de pré-processamento (fit no treino, transform nos demais).
from src.data.preprocessor import preparar_features
# Importa hiperparâmetros e diretórios centrais do projeto.
from src.utils.config import LEARNING_RATE, EPOCHS, BATCH_SIZE, EXPERIMENTS_DIR, IMAGES_DIR

# Define as três arquiteturas comparadas, em ordem crescente de complexidade.
ARQUITETURAS = {'32': [32], '64→32': [64, 32], '128→64→32': [128, 64, 32]}
# Define as seeds da avaliação principal (0 a 4).
SEEDS = (0, 1, 2, 3, 4)
# Define os critérios de early stopping comparados.
MONITORES = ('val_loss', 'val_f1')
# Associa a cada critério o sentido da otimização.
MODO_POR_MONITOR = {'val_loss': 'min', 'val_f1': 'max'}
# Define a melhora mínima exigida pelo early stopping (igual ao notebook original).
MIN_DELTA = 1e-4
# Define a paciência do early stopping (igual ao notebook original).
PACIENCIA = 10
# Define o teto de épocas (50, de config.py, igual ao notebook original).
EPOCAS_MAX = EPOCHS
# Define o lote de treino (32, de config.py, igual ao notebook original).
LOTE_TREINO = BATCH_SIZE
# Define o lote de avaliação, maior para acelerar; não altera as métricas.
LOTE_AVALIACAO = 4096
# Define o decaimento de pesos base (igual ao notebook original).
WEIGHT_DECAY_BASE = 1e-4
# Define o dropout base (igual ao notebook original).
DROPOUT_BASE = 0.2
# Define a pasta de saída dos CSVs e históricos desta análise.
DIR_RESULTADOS = EXPERIMENTS_DIR / 'selecao_arquitetura'
# Define a pasta de saída das figuras.
DIR_FIGURAS = IMAGES_DIR
# Define as colunas do CSV de resultados por execução.
COLUNAS_CSV = [
    # Identificação da execução.
    'arquitetura', 'monitor', 'seed', 'melhor_epoca', 'epocas_treinadas',
    # Métricas de validação da melhor época (pesos restaurados).
    'val_loss', 'val_acc', 'val_prec', 'val_rec', 'val_f1',
    # Métricas do teste 2025 com os mesmos pesos (apenas confirmação).
    'test_loss', 'test_acc', 'test_prec', 'test_rec', 'test_f1',
    # Custo do modelo.
    'n_parametros', 'tempo_treino_s',
]


class MLP(nn.Module):
    """MLP idêntica à do notebook: Linear → ReLU → Dropout por camada oculta, saída linear (logit)."""

    def __init__(self, input_size, hidden_sizes, dropout=0.2):
        # Inicializa a classe base do PyTorch.
        super().__init__()
        # Cria a lista de camadas da rede.
        camadas = []
        # Começa com a dimensão de entrada (número de features).
        dim_anterior = input_size
        # Percorre os tamanhos das camadas ocultas.
        for tamanho in hidden_sizes:
            # Adiciona camada linear, ativação ReLU e dropout, na mesma ordem do notebook.
            camadas += [nn.Linear(dim_anterior, tamanho), nn.ReLU(), nn.Dropout(dropout)]
            # Atualiza a dimensão de entrada da próxima camada.
            dim_anterior = tamanho
        # Adiciona a camada de saída com 1 logit (a sigmoide fica na BCEWithLogitsLoss).
        camadas.append(nn.Linear(dim_anterior, 1))
        # Encadeia as camadas em um módulo sequencial.
        self.rede = nn.Sequential(*camadas)

    def forward(self, x):
        # Propaga a entrada pela rede e devolve o logit.
        return self.rede(x)


class EarlyStoppingConfiguravel:
    """Early stopping por métrica: monitor='val_loss'|'val_f1', mode='min'|'max'."""

    def __init__(self, patience=PACIENCIA, min_delta=MIN_DELTA, monitor='val_loss', mode='min'):
        # Rejeita combinações inválidas de modo.
        if mode not in ('min', 'max'):
            # Interrompe com mensagem clara se o modo for desconhecido.
            raise ValueError("mode deve ser 'min' ou 'max'")
        # Guarda a paciência (épocas sem melhora toleradas).
        self.patience = patience
        # Guarda a melhora mínima exigida.
        self.min_delta = min_delta
        # Guarda o nome da métrica monitorada.
        self.monitor = monitor
        # Guarda o sentido da otimização.
        self.mode = mode
        # Inicializa o melhor valor como o pior possível para o modo escolhido.
        self.best = float('inf') if mode == 'min' else float('-inf')
        # Inicializa o contador de épocas sem melhora.
        self.counter = 0
        # Inicializa a época do melhor valor (0 = nenhuma ainda).
        self.best_epoch = 0
        # Inicializa a cópia dos melhores pesos.
        self.best_state = None

    def melhorou(self, valor):
        # Em 'min', melhora significa cair mais que min_delta.
        if self.mode == 'min':
            # Devolve True se o valor ficou abaixo do melhor menos a tolerância.
            return valor < self.best - self.min_delta
        # Em 'max', melhora significa subir mais que min_delta.
        return valor > self.best + self.min_delta

    def __call__(self, valor, modelo, epoca):
        # Verifica se a métrica desta época é a melhor até agora.
        if self.melhorou(valor):
            # Atualiza o melhor valor.
            self.best = valor
            # Registra a época do melhor valor.
            self.best_epoch = epoca
            # Zera o contador de paciência.
            self.counter = 0
            # Copia os pesos atuais para a CPU (restauração posterior).
            self.best_state = {k: v.detach().cpu().clone() for k, v in modelo.state_dict().items()}
            # Sinaliza para continuar o treino.
            return False
        # Sem melhora: incrementa o contador.
        self.counter += 1
        # Sinaliza parada quando a paciência se esgota.
        return self.counter >= self.patience

    def restore(self, modelo):
        # Só restaura se algum estado tiver sido salvo.
        if self.best_state is not None:
            # Carrega no modelo os pesos da melhor época.
            modelo.load_state_dict(self.best_state)


def fixar_sementes(seed):
    """Fixa random, numpy e torch; na GPU, ativa cuDNN e algoritmos determinísticos."""
    # Fixa a semente do gerador padrão do Python.
    random.seed(seed)
    # Fixa a semente do numpy.
    np.random.seed(seed)
    # Fixa a semente do torch na CPU e em todas as GPUs.
    torch.manual_seed(seed)
    # Verifica se há GPU disponível.
    if torch.cuda.is_available():
        # Fixa a semente de todas as GPUs explicitamente.
        torch.cuda.manual_seed_all(seed)
        # Obriga o cuDNN a usar algoritmos determinísticos.
        torch.backends.cudnn.deterministic = True
        # Desliga a escolha automática (não determinística) de algoritmos do cuDNN.
        torch.backends.cudnn.benchmark = False
    # Obriga o PyTorch a usar apenas operações determinísticas (avisa em vez de falhar se faltar alguma).
    torch.use_deterministic_algorithms(True, warn_only=True)


def carregar_dados_2025():
    """Carrega 2025, divide 70/15/15 e pré-processa exatamente como o notebook."""
    # Informa o início do carregamento.
    print('Carregando dados 2025...')
    # Lê os microdados de 2025 já filtrados para o município e com alvo definido.
    df_2025 = carregar_e_preparar(ano=2025)
    # Consolida boletins e divide em treino, validação e teste 2025 (seed 42 do config).
    treino, val, teste = preparar_split_2025(df_2025)
    # Ajusta o pré-processador somente no treino e transforma o treino.
    X_tr, y_tr, prep = preparar_features(treino, fit=True)
    # Transforma a validação com o pré-processador do treino.
    X_va, y_va, _ = preparar_features(val, preprocessor=prep, fit=False)
    # Transforma o teste 2025 com o pré-processador do treino.
    X_te, y_te, _ = preparar_features(teste, preprocessor=prep, fit=False)
    # Mostra as dimensões para conferência com o notebook (1914 features).
    print(f'Treino {X_tr.shape} | Val {X_va.shape} | Teste2025 {X_te.shape}')
    # Devolve os seis arrays.
    return X_tr, y_tr, X_va, y_va, X_te, y_te


def para_tensor(X, y, dispositivo):
    """Converte um par (X, y) em tensores float32 residentes no dispositivo."""
    # Converte matriz esparsa em densa, se o pré-processador devolver esparsa.
    X = X.toarray() if hasattr(X, 'toarray') else np.asarray(X)
    # Cria o tensor de entradas no dispositivo.
    X_t = torch.tensor(X, dtype=torch.float32, device=dispositivo)
    # Cria o tensor de rótulos no dispositivo.
    y_t = torch.tensor(np.asarray(y), dtype=torch.float32, device=dispositivo)
    # Devolve o par de tensores.
    return X_t, y_t


def treinar_epoca(modelo, X, y, perda_fn, otimizador, gerador):
    """Uma época com embaralhamento por seed e lotes de 32, tudo na GPU."""
    # Ativa o modo de treino (dropout ligado).
    modelo.train()
    # Obtém o número de amostras de treino.
    n = X.shape[0]
    # Gera uma permutação reprodutível dos índices (equivale ao shuffle=True do DataLoader).
    ordem = torch.randperm(n, generator=gerador, device=X.device)
    # Acumula a perda no dispositivo para evitar sincronizar a cada lote.
    soma = torch.zeros((), device=X.device)
    # Percorre os lotes.
    for inicio in range(0, n, LOTE_TREINO):
        # Seleciona os índices do lote atual.
        idx = ordem[inicio:inicio + LOTE_TREINO]
        # Calcula os logits do lote.
        logits = modelo(X[idx])
        # Calcula a BCE com logits contra os rótulos do lote.
        perda = perda_fn(logits, y[idx].unsqueeze(1))
        # Zera os gradientes acumulados.
        otimizador.zero_grad(set_to_none=True)
        # Calcula os gradientes.
        perda.backward()
        # Atualiza os pesos.
        otimizador.step()
        # Soma a perda ponderada pelo tamanho do lote.
        soma += perda.detach() * idx.numel()
    # Devolve a perda média da época.
    return (soma / n).item()


@torch.no_grad()
def avaliar(modelo, X, y, perda_fn):
    """Perda, acurácia, precisão, recall e F1 macro (mesmas definições do notebook)."""
    # Ativa o modo de avaliação (dropout desligado).
    modelo.eval()
    # Calcula os logits por blocos e concatena.
    logits = torch.cat([modelo(X[i:i + LOTE_AVALIACAO]) for i in range(0, X.shape[0], LOTE_AVALIACAO)])
    # Calcula a perda média no conjunto inteiro.
    perda = perda_fn(logits, y.unsqueeze(1)).item()
    # Converte logits em classe com limiar 0,5 (logit > 0) e leva para a CPU.
    pred = (logits.squeeze(1) > 0).float().cpu().numpy()
    # Leva os rótulos verdadeiros para a CPU.
    real = y.cpu().numpy()
    # Devolve as métricas: precisão e recall da classe roubo; F1 macro, como no notebook.
    return {
        # Perda BCE média.
        'loss': perda,
        # Acurácia.
        'acc': accuracy_score(real, pred),
        # Precisão da classe 1 (roubo).
        'prec': precision_score(real, pred, zero_division=0),
        # Recall da classe 1 (roubo).
        'rec': recall_score(real, pred, zero_division=0),
        # F1 macro (média das duas classes).
        'f1': f1_score(real, pred, average='macro', zero_division=0),
    }


def treinar_execucao(nome, ocultas, monitor, seed, dados, dispositivo,
                     lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY_BASE, dropout=DROPOUT_BASE):
    """Treina uma configuração com uma seed; devolve (linha de resultados, histórico)."""
    # Desempacota os tensores de treino, validação e teste 2025.
    X_tr, y_tr, X_va, y_va, X_te, y_te = dados
    # Fixa todas as sementes antes de criar o modelo (inicialização reprodutível).
    fixar_sementes(seed)
    # Cria a rede e move para o dispositivo.
    modelo = MLP(X_tr.shape[1], ocultas, dropout=dropout).to(dispositivo)
    # Define a perda BCE com sigmoide integrada.
    perda_fn = nn.BCEWithLogitsLoss()
    # Cria o otimizador AdamW com os hiperparâmetros da execução.
    otimizador = AdamW(modelo.parameters(), lr=lr, weight_decay=weight_decay)
    # Cria o early stopping no critério pedido.
    early = EarlyStoppingConfiguravel(monitor=monitor, mode=MODO_POR_MONITOR[monitor])
    # Cria um gerador próprio para o embaralhamento, ligado à seed.
    gerador = torch.Generator(device=dispositivo)
    # Fixa a semente do gerador de embaralhamento.
    gerador.manual_seed(seed)
    # Inicializa o histórico por época.
    hist = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}
    # Marca o início do treino.
    t0 = time.perf_counter()
    # Percorre as épocas até o teto.
    for epoca in range(1, EPOCAS_MAX + 1):
        # Treina uma época e obtém a perda de treino.
        perda_tr = treinar_epoca(modelo, X_tr, y_tr, perda_fn, otimizador, gerador)
        # Avalia na validação.
        m_va = avaliar(modelo, X_va, y_va, perda_fn)
        # Registra a perda de treino.
        hist['train_loss'].append(perda_tr)
        # Registra a perda de validação.
        hist['val_loss'].append(m_va['loss'])
        # Registra a acurácia de validação.
        hist['val_acc'].append(m_va['acc'])
        # Registra o F1 de validação.
        hist['val_f1'].append(m_va['f1'])
        # Escolhe o valor monitorado conforme o critério.
        valor = m_va['loss'] if monitor == 'val_loss' else m_va['f1']
        # Atualiza o early stopping e para se a paciência acabou.
        if early(valor, modelo, epoca):
            # Sai do laço de épocas.
            break
    # Mede o tempo total de treino.
    tempo = time.perf_counter() - t0
    # Restaura os pesos da melhor época.
    early.restore(modelo)
    # Reavalia a validação com os pesos restaurados (todas as métricas da melhor época).
    m_va = avaliar(modelo, X_va, y_va, perda_fn)
    # Avalia o teste 2025 com os mesmos pesos (apenas para confirmação posterior).
    m_te = avaliar(modelo, X_te, y_te, perda_fn)
    # Monta a linha de resultados da execução.
    linha = {
        # Nome da arquitetura.
        'arquitetura': nome,
        # Critério de early stopping.
        'monitor': monitor,
        # Seed usada.
        'seed': seed,
        # Época dos pesos restaurados.
        'melhor_epoca': early.best_epoch,
        # Número de épocas efetivamente treinadas.
        'epocas_treinadas': len(hist['train_loss']),
        # Métricas de validação da melhor época, com prefixo val_.
        **{f'val_{k}': v for k, v in m_va.items()},
        # Métricas do teste 2025, com prefixo test_.
        **{f'test_{k}': v for k, v in m_te.items()},
        # Número de parâmetros treináveis.
        'n_parametros': sum(p.numel() for p in modelo.parameters() if p.requires_grad),
        # Tempo de treino em segundos.
        'tempo_treino_s': tempo,
    }
    # Devolve a linha e o histórico por época.
    return linha, hist


def preparar_dispositivo_e_dados():
    """Escolhe o dispositivo e carrega os dados de 2025 como tensores."""
    # Usa GPU se disponível, senão CPU.
    dispositivo = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Informa o dispositivo.
    print(f'Dispositivo: {dispositivo}')
    # Carrega e pré-processa 2025.
    X_tr, y_tr, X_va, y_va, X_te, y_te = carregar_dados_2025()
    # Converte o treino em tensores no dispositivo.
    X_tr, y_tr = para_tensor(X_tr, y_tr, dispositivo)
    # Converte a validação em tensores no dispositivo.
    X_va, y_va = para_tensor(X_va, y_va, dispositivo)
    # Converte o teste 2025 em tensores no dispositivo.
    X_te, y_te = para_tensor(X_te, y_te, dispositivo)
    # Devolve o dispositivo e a tupla de tensores.
    return dispositivo, (X_tr, y_tr, X_va, y_va, X_te, y_te)


def executar_etapa_seeds():
    """Etapa 5: 3 arquiteturas × 2 critérios × 5 seeds; grava CSV e históricos."""
    # Prepara dispositivo e dados.
    dispositivo, dados = preparar_dispositivo_e_dados()
    # Cria a pasta de resultados se não existir.
    DIR_RESULTADOS.mkdir(parents=True, exist_ok=True)
    # Define o caminho do CSV de resultados.
    caminho_csv = DIR_RESULTADOS / 'resultados_seeds.csv'
    # Define o caminho do JSON de históricos.
    caminho_hist = DIR_RESULTADOS / 'historicos_seeds.json'
    # Inicializa a lista de linhas.
    linhas = []
    # Inicializa o dicionário de históricos.
    historicos = {}
    # Retomada: se o CSV já existe, reaproveita as execuções concluídas antes de uma interrupção.
    if caminho_csv.exists():
        # Lê o CSV parcial e converte cada linha em dicionário.
        linhas = pd.read_csv(caminho_csv).to_dict('records')
    # Retomada: se o JSON de históricos já existe, reaproveita os históricos salvos.
    if caminho_hist.exists():
        # Lê os históricos por época já gravados.
        historicos = json.loads(caminho_hist.read_text(encoding='utf-8'))
    # Monta o conjunto (arquitetura, critério, seed) das execuções que têm resultado no CSV e histórico no JSON.
    feitas_csv = {(l['arquitetura'], l['monitor'], int(l['seed'])) for l in linhas
                  # Só conta como feita se o histórico por época também existir (necessário para as curvas).
                  if f"{l['arquitetura']}|{l['monitor']}|{int(l['seed'])}" in historicos}
    # Descarta linhas sem histórico (serão refeitas) para não duplicar no CSV.
    linhas = [l for l in linhas if (l['arquitetura'], l['monitor'], int(l['seed'])) in feitas_csv]
    # Informa quantas execuções foram reaproveitadas (não serão treinadas de novo).
    print(f'Retomada: {len(feitas_csv)} execucoes reaproveitadas', flush=True)
    # Calcula o total de execuções para o log de progresso.
    total = len(ARQUITETURAS) * len(MONITORES) * len(SEEDS)
    # Percorre arquiteturas, critérios e seeds.
    for i, (nome, ocultas, monitor, seed) in enumerate(
            # Gera todas as combinações em ordem fixa.
            [(n, o, m, s) for n, o in ARQUITETURAS.items() for m in MONITORES for s in SEEDS], start=1):
        # Pula combinações já concluídas (retomada após interrupção).
        if (nome, monitor, seed) in feitas_csv:
            # Vai para a próxima combinação.
            continue
        # Treina a execução.
        linha, hist = treinar_execucao(nome, ocultas, monitor, seed, dados, dispositivo)
        # Guarda a linha.
        linhas.append(linha)
        # Guarda o histórico com chave legível.
        historicos[f'{nome}|{monitor}|{seed}'] = hist
        # Regrava o CSV a cada execução para não perder progresso.
        pd.DataFrame(linhas, columns=COLUNAS_CSV).to_csv(caminho_csv, index=False)
        # Regrava os históricos a cada execução, pelo mesmo motivo.
        caminho_hist.write_text(json.dumps(historicos, ensure_ascii=False), encoding='utf-8')
        # Mostra o progresso usando só ASCII (o console cp1252 do Windows não codifica '→').
        print(f"[{i}/{total}] {nome.replace('→', '-'):>10} | {monitor:8} | seed {seed} | melhor ep. "
              # Continua a mensagem com época, métrica e tempo.
              f"{linha['melhor_epoca']:2d} | val_f1 {linha['val_f1']:.4f} | {linha['tempo_treino_s']:.0f}s", flush=True)
    # Informa onde os arquivos foram salvos.
    print(f'CSV: {caminho_csv}\nHistóricos: {caminho_hist}')
    # Devolve o DataFrame de resultados e os históricos.
    return pd.DataFrame(linhas, columns=COLUNAS_CSV), historicos


def tabela_media_desvio(df, metricas=('melhor_epoca', 'val_loss', 'val_acc', 'val_f1',
                                      'test_loss', 'test_acc', 'test_prec', 'test_rec', 'test_f1',
                                      'n_parametros', 'tempo_treino_s')):
    """Etapa 6: tabela 'média ± desvio' por arquitetura e critério."""
    # Agrupa por critério e arquitetura e calcula média e desvio amostral.
    agg = df.groupby(['monitor', 'arquitetura'], sort=False)[list(metricas)].agg(['mean', 'std'])
    # Cria a tabela de saída com o mesmo índice.
    saida = pd.DataFrame(index=agg.index)
    # Percorre as métricas pedidas.
    for m in metricas:
        # Usa 0 casas para inteiros/tempo e 4 para as demais.
        casas = 0 if m in ('n_parametros',) else (1 if m in ('melhor_epoca', 'tempo_treino_s') else 4)
        # Formata 'média ± desvio' como texto.
        saida[m] = [f'{a:.{casas}f} ± {b:.{casas}f}' for a, b in zip(agg[(m, 'mean')], agg[(m, 'std')])]
    # Devolve a tabela formatada.
    return saida


def testes_pareados(df, metrica='val_f1'):
    """Etapa 6: t pareado e Wilcoxon por seed para cada par de arquiteturas (n=5, baixo poder)."""
    # Lista os nomes em ordem de complexidade.
    nomes = list(ARQUITETURAS)
    # Inicializa a lista de resultados.
    linhas = []
    # Percorre os critérios de early stopping.
    for monitor in MONITORES:
        # Monta tabela seed × arquitetura com a métrica (pareamento por seed).
        piv = df[df.monitor == monitor].pivot(index='seed', columns='arquitetura', values=metrica)
        # Percorre todos os pares distintos de arquiteturas.
        for i in range(len(nomes)):
            # Percorre o segundo elemento do par.
            for j in range(i + 1, len(nomes)):
                # Obtém os nomes do par.
                a, b = nomes[i], nomes[j]
                # Calcula a diferença por seed (b − a).
                d = piv[b] - piv[a]
                # Executa o teste t pareado.
                t = stats.ttest_rel(piv[b], piv[a])
                # Executa Wilcoxon; com n=5 o menor p bilateral possível é 0,0625.
                w = stats.wilcoxon(piv[b], piv[a]) if (d != 0).any() else None
                # Registra o resultado do par.
                linhas.append({
                    # Critério de early stopping.
                    'monitor': monitor,
                    # Par comparado.
                    'comparacao': f'{b} − {a}',
                    # Diferença média.
                    'dif_media': d.mean(),
                    # Desvio da diferença.
                    'dif_desvio': d.std(),
                    # Seeds em que b superou a.
                    'b_vence_em': f'{int((d > 0).sum())}/{len(d)}',
                    # p-valor do t pareado.
                    'p_t_pareado': t.pvalue,
                    # p-valor do Wilcoxon (NaN se todas as diferenças forem zero).
                    'p_wilcoxon': w.pvalue if w is not None else np.nan,
                })
    # Devolve a tabela de testes.
    return pd.DataFrame(linhas)


def regra_decisao(df, monitor='val_loss'):
    """Etapa 7: maior val_F1 médio, salvo se uma mais simples estiver a ≤ 1 desvio → a mais simples."""
    # Filtra o critério de early stopping.
    sub = df[df.monitor == monitor]
    # Calcula média, desvio e parâmetros por arquitetura.
    est = sub.groupby('arquitetura').agg(media=('val_f1', 'mean'), desvio=('val_f1', 'std'),
                                         # Número de parâmetros (constante por arquitetura).
                                         params=('n_parametros', 'first'))
    # Ordena da mais simples para a mais complexa.
    est = est.sort_values('params')
    # Identifica a arquitetura de maior val_F1 médio.
    melhor = est['media'].idxmax()
    # Percorre as arquiteturas da mais simples para a mais complexa.
    for nome, row in est.iterrows():
        # Para ao chegar na própria melhor (nenhuma mais simples a substituiu).
        if nome == melhor:
            # Interrompe a busca.
            break
        # Calcula a diferença de média para a melhor.
        dif = est.loc[melhor, 'media'] - row['media']
        # Usa como tolerância o maior desvio entre as duas (critério conservador).
        tol = max(est.loc[melhor, 'desvio'], row['desvio'])
        # Se a diferença está dentro de 1 desvio, escolhe a mais simples.
        if dif <= tol:
            # Monta a justificativa.
            motivo = (f'{melhor} tem o maior val_F1 médio ({est.loc[melhor, "media"]:.4f}), mas {nome} '
                      # Continua a justificativa com a diferença e a tolerância.
                      f'fica a {dif:.4f}, dentro de 1 desvio ({tol:.4f}); por parcimônia escolhe-se {nome}.')
            # Devolve a escolha, a tabela e o motivo.
            return nome, est, motivo
    # Caso nenhuma mais simples esteja dentro da tolerância, mantém a melhor.
    return melhor, est, f'{melhor} tem o maior val_F1 médio e nenhuma mais simples fica a ≤ 1 desvio.'


def figura_curvas(historicos, monitor='val_loss', seed=0, caminho=None):
    """Etapa 3: perda (melhor época marcada) e Val F1 por época, uma coluna por arquitetura."""
    # Cria grade 2 × 3: perda em cima, Val F1 embaixo.
    fig, eixos = plt.subplots(2, len(ARQUITETURAS), figsize=(16, 8), sharex='col')
    # Percorre as arquiteturas.
    for c, nome in enumerate(ARQUITETURAS):
        # Obtém o histórico da execução de referência.
        h = historicos[f'{nome}|{monitor}|{seed}']
        # Gera o eixo de épocas a partir de 1.
        ep = np.arange(1, len(h['train_loss']) + 1)
        # Localiza a época de menor val_loss.
        i_min = int(np.argmin(h['val_loss']))
        # Converte o índice em número de época.
        ep_min = i_min + 1
        # Seleciona o painel de perda.
        ax = eixos[0, c]
        # Plota a perda de treino.
        ax.plot(ep, h['train_loss'], color='tab:blue', lw=2, label='Treino')
        # Plota a perda de validação.
        ax.plot(ep, h['val_loss'], color='tab:red', lw=2, label='Validação')
        # Desenha a linha tracejada na melhor época, com a época na legenda.
        ax.axvline(ep_min, color='gray', ls='--', lw=1.2, label=f'Melhor época = {ep_min}')
        # Marca o ponto mínimo da val_loss.
        ax.plot(ep_min, h['val_loss'][i_min], 'o', color='black', ms=7,
                # Rótulo do ponto com o valor mínimo.
                label=f'Mín. val_loss = {h["val_loss"][i_min]:.4f}')
        # Define o título da coluna.
        ax.set_title(f'Arquitetura {nome}', fontweight='bold')
        # Define o rótulo do eixo Y.
        ax.set_ylabel('Perda (BCE)')
        # Mostra a legenda.
        ax.legend(fontsize=8)
        # Adiciona grade leve.
        ax.grid(alpha=0.3)
        # Seleciona o painel de Val F1.
        ax2 = eixos[1, c]
        # Plota o F1 macro de validação por época.
        ax2.plot(ep, h['val_f1'], color='tab:green', lw=2, marker='.', label='Val F1 (macro)')
        # Repete a linha da melhor época (por val_loss).
        ax2.axvline(ep_min, color='gray', ls='--', lw=1.2, label=f'Melhor época = {ep_min}')
        # Marca o Val F1 na melhor época.
        ax2.plot(ep_min, h['val_f1'][i_min], 'o', color='black', ms=7)
        # Define o rótulo do eixo X.
        ax2.set_xlabel('Época')
        # Define o rótulo do eixo Y.
        ax2.set_ylabel('Val F1 macro')
        # Mostra a legenda.
        ax2.legend(fontsize=8)
        # Adiciona grade leve.
        ax2.grid(alpha=0.3)
    # Define o título geral.
    fig.suptitle(f'Curvas por época (early stopping por {monitor}, seed {seed})', fontweight='bold')
    # Ajusta o espaçamento.
    fig.tight_layout()
    # Salva a figura se um caminho foi informado.
    if caminho is not None:
        # Grava o PNG.
        fig.savefig(caminho, dpi=150, bbox_inches='tight')
    # Devolve a figura.
    return fig


def figura_comparacao(df, caminho=None):
    """Etapa 6: boxplot + pontos por seed de val_F1 e test_F1, por arquitetura e critério."""
    # Cria grade 2 × 2: linhas = critério, colunas = métrica.
    fig, eixos = plt.subplots(len(MONITORES), 2, figsize=(13, 9), sharey='row')
    # Lista as arquiteturas em ordem de complexidade.
    nomes = list(ARQUITETURAS)
    # Percorre os critérios.
    for r, monitor in enumerate(MONITORES):
        # Percorre as métricas.
        for c, (col, titulo) in enumerate([('val_f1', 'Val F1 (seleção)'), ('test_f1', 'Test F1 2025 (confirmação)')]):
            # Seleciona o painel.
            ax = eixos[r, c]
            # Reúne os valores por arquitetura.
            valores = [df[(df.monitor == monitor) & (df.arquitetura == n)][col].values for n in nomes]
            # Desenha os boxplots.
            ax.boxplot(valores, tick_labels=nomes, widths=0.5, showmeans=True)
            # Percorre as arquiteturas para sobrepor os pontos.
            for k, v in enumerate(valores, start=1):
                # Plota cada seed como um ponto com leve deslocamento horizontal.
                ax.scatter(np.full(len(v), k) + np.linspace(-0.08, 0.08, len(v)), v, s=25, alpha=0.8, zorder=3)
            # Define o título do painel.
            ax.set_title(f'{titulo} — ES por {monitor}', fontsize=10)
            # Adiciona grade leve no eixo Y.
            ax.grid(axis='y', alpha=0.3)
    # Define o título geral.
    fig.suptitle('Comparação entre arquiteturas — 5 seeds (triângulo = média)', fontweight='bold')
    # Ajusta o espaçamento.
    fig.tight_layout()
    # Salva a figura se um caminho foi informado.
    if caminho is not None:
        # Grava o PNG.
        fig.savefig(caminho, dpi=150, bbox_inches='tight')
    # Devolve a figura.
    return fig


def relatorio(df, historicos):
    """Imprime tabelas, testes e decisão; salva as figuras."""
    # Configura o pandas para mostrar todas as colunas.
    pd.set_option('display.width', 250)
    # Imprime a tabela média ± desvio.
    print('\n=== Média ± desvio (5 seeds) ===\n', tabela_media_desvio(df).to_string())
    # Imprime os testes pareados.
    print('\n=== Testes pareados de val_F1 (n=5) ===\n', testes_pareados(df).round(4).to_string(index=False))
    # Percorre os critérios para aplicar a regra de decisão.
    for monitor in MONITORES:
        # Aplica a regra de decisão.
        escolhida, _, motivo = regra_decisao(df, monitor)
        # Imprime a decisão.
        print(f'\n[Decisão | ES por {monitor}] {escolhida}: {motivo}')
    # Cria a pasta de figuras se não existir.
    DIR_FIGURAS.mkdir(parents=True, exist_ok=True)
    # Salva as curvas de perda.
    figura_curvas(historicos, caminho=DIR_FIGURAS / 'curvas_perda.png')
    # Salva a comparação entre seeds.
    figura_comparacao(df, caminho=DIR_FIGURAS / 'comparacao_seeds.png')
    # Informa onde as figuras foram salvas.
    print(f'\nFiguras em {DIR_FIGURAS}')


def executar_etapa_opcional(seeds=(0, 1, 2)):
    """Etapa 8: 128→64→32 com wd ∈ {1e-4,1e-3} × dropout ∈ {0,3; 0,5} × lr ∈ {1e-4, 5e-5}."""
    # Prepara dispositivo e dados.
    dispositivo, dados = preparar_dispositivo_e_dados()
    # Inicializa a lista de linhas.
    linhas = []
    # Percorre os valores de weight decay.
    for wd in (1e-4, 1e-3):
        # Percorre os valores de dropout.
        for dp in (0.3, 0.5):
            # Percorre a lr atual e sua metade.
            for lr in (LEARNING_RATE, LEARNING_RATE / 2):
                # Percorre as seeds.
                for seed in seeds:
                    # Treina a configuração com early stopping por val_loss (padrão).
                    linha, _ = treinar_execucao('128→64→32', ARQUITETURAS['128→64→32'], 'val_loss', seed,
                                                # Passa dados, dispositivo e hiperparâmetros da grade.
                                                dados, dispositivo, lr=lr, weight_decay=wd, dropout=dp)
                    # Anota os hiperparâmetros na linha.
                    linha.update(weight_decay=wd, dropout=dp, lr=lr)
                    # Guarda a linha.
                    linhas.append(linha)
                    # Mostra o progresso.
                    print(f"wd={wd:g} dp={dp} lr={lr:g} seed={seed} | ép {linha['melhor_epoca']} "
                          # Continua a mensagem com o F1.
                          f"| val_f1 {linha['val_f1']:.4f}", flush=True)
                    # Regrava o CSV a cada execução.
                    pd.DataFrame(linhas).to_csv(DIR_RESULTADOS / 'resultados_opcional.csv', index=False)
    # Converte em DataFrame.
    df = pd.DataFrame(linhas)
    # Resume val_F1 por combinação.
    resumo = df.groupby(['weight_decay', 'dropout', 'lr'])['val_f1'].agg(['mean', 'std', 'min', 'max'])
    # Imprime o resumo ordenado pela média.
    print('\n=== Etapa 8: val_F1 por combinação ===\n', resumo.sort_values('mean', ascending=False).round(4))
    # Devolve o DataFrame.
    return df


# Executa apenas quando chamado como script.
if __name__ == '__main__':
    # Cria o leitor de argumentos.
    parser = argparse.ArgumentParser(description='Seleção de arquitetura MLP com várias seeds.')
    # Declara o argumento --etapa.
    parser.add_argument('--etapa', choices=['seeds', 'relatorio', 'opcional'], default='seeds')
    # Lê os argumentos.
    args = parser.parse_args()
    # Etapa principal: treina e gera o relatório.
    if args.etapa == 'seeds':
        # Treina as 30 execuções.
        df_res, hist_res = executar_etapa_seeds()
        # Gera tabelas, testes, decisão e figuras.
        relatorio(df_res, hist_res)
    # Só refaz o relatório a partir dos arquivos já salvos.
    elif args.etapa == 'relatorio':
        # Lê o CSV de resultados.
        df_res = pd.read_csv(DIR_RESULTADOS / 'resultados_seeds.csv')
        # Abre o JSON de históricos.
        with open(DIR_RESULTADOS / 'historicos_seeds.json', encoding='utf-8') as f:
            # Carrega os históricos.
            hist_res = json.load(f)
        # Gera tabelas, testes, decisão e figuras.
        relatorio(df_res, hist_res)
    # Etapa opcional (8).
    else:
        # Roda a grade da rede maior.
        executar_etapa_opcional()
