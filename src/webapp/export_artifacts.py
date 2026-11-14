"""Exporta o modelo MLP escolhido e o pré-processador para a aplicação web no Netlify.

O Netlify Functions não executa Python nem PyTorch; por isso, os pesos da rede e os
parâmetros do pré-processamento são gravados como módulos JavaScript, e a inferência
é refeita em JavaScript puro em netlify/functions/classificar.mjs.

Saídas:
- netlify/functions/artefatos/pesos.mjs        pesos e vieses das camadas lineares (64→32→1)
- netlify/functions/artefatos/preprocesso.mjs  categorias do OneHotEncoder, medianas e scaler
- netlify/functions/artefatos/metadados.mjs    arquitetura, limiar, métricas e versão do modelo
- netlify/public/data/opcoes.json              opções do formulário (somente do treino 2025)
- netlify/functions/artefatos/casos_teste.json casos de validação para conferir a paridade Python × JS

Execução (na raiz do projeto): python -m src.webapp.export_artifacts
Pré-requisito: models/best/ gerado pelo notebook 03_treinamento_MLP.ipynb e data/raw/ com o xlsx de 2025.
"""

# Importa hashlib para calcular a impressão digital (SHA-256) do arquivo de pesos.
import hashlib
# Importa json para gravar os artefatos em formato texto.
import json
# Importa datetime e timezone para registrar a data da exportação em UTC.
from datetime import datetime, timezone
# Importa Path para montar caminhos de arquivos de forma portátil.
from pathlib import Path

# Importa joblib para ler o pré-processador serializado (preprocessor.pkl).
import joblib
# Importa numpy para converter arrays em listas.
import numpy as np
# Importa torch para ler o state_dict do modelo (mlp_model.pth) e calcular as saídas de referência.
import torch

# Importa o carregador dos microdados (filtro São Paulo e alvo roubo/furto).
from src.data.loader import carregar_e_preparar
# Importa a função de pré-processamento usada no treino (mesma transformação da API).
from src.data.preprocessor import CATEGORICAS_NOMINAIS, NUMERICAS, preparar_features
# Importa a consolidação por boletim e a divisão 70/15/15 usadas no treino.
from src.data.splitter import preparar_split_2025

# Define a raiz do repositório (dois níveis acima de src/webapp/).
RAIZ = Path(__file__).resolve().parents[2]
# Define a pasta com os artefatos do modelo escolhido.
DIR_MODELO = RAIZ / 'models' / 'best'
# Define a pasta de saída dos módulos usados pela função serverless.
DIR_ARTEFATOS = RAIZ / 'netlify' / 'functions' / 'artefatos'
# Define a pasta de dados estáticos lidos pelo navegador.
DIR_DADOS_PUBLICOS = RAIZ / 'netlify' / 'public' / 'data'
# Define o número máximo de bairros listados no formulário (os demais podem ser digitados).
MAX_BAIRROS = 400
# Define o número máximo de subtipos sugeridos para cada tipo de local.
MAX_SUBTIPOS = 25
# Define quantos casos do conjunto de validação são gravados para o teste de paridade.
N_CASOS_TESTE = 200


def gravar_modulo(caminho, objeto):
    """Grava `objeto` como módulo ES (`export default ...`) em `caminho`."""
    # Cria a pasta de destino caso ainda não exista.
    caminho.parent.mkdir(parents=True, exist_ok=True)
    # Monta o cabeçalho que avisa que o arquivo é gerado automaticamente.
    cabecalho = '// Gerado por src/webapp/export_artifacts.py. Não editar à mão.\n'
    # Grava o módulo com o objeto serializado em JSON compacto e UTF-8.
    caminho.write_text(cabecalho + 'export default ' + json.dumps(objeto, ensure_ascii=False, separators=(',', ':')) + ';\n', encoding='utf-8')
    # Informa no console o arquivo gravado e seu tamanho em kB.
    print(f'  gravado {caminho.relative_to(RAIZ)} ({caminho.stat().st_size / 1024:.0f} kB)')


def exportar_pesos(metadados):
    """Converte o state_dict da MLP em listas de pesos e vieses por camada."""
    # Lê o arquivo de pesos em bytes para calcular a versão do modelo.
    bytes_pth = (DIR_MODELO / 'mlp_model.pth').read_bytes()
    # Usa os 12 primeiros caracteres do SHA-256 como identificador da versão.
    versao = hashlib.sha256(bytes_pth).hexdigest()[:12]
    # Carrega o state_dict na CPU (a exportação não precisa de GPU).
    estado = torch.load(DIR_MODELO / 'mlp_model.pth', map_location='cpu', weights_only=True)
    # Lista os nomes das camadas na ordem do forward: ocultas e, por último, a saída.
    nomes = [f'layers.{i}' for i in range(len(metadados['hidden_sizes']))] + ['output_layer']
    # Converte cada camada em {W: matriz [saída][entrada], b: vetor}, com 7 algarismos significativos.
    camadas = [{'W': np.round(estado[f'{n}.weight'].numpy().astype(float), 7).tolist(),
                # Converte o vetor de vieses da mesma camada.
                'b': np.round(estado[f'{n}.bias'].numpy().astype(float), 7).tolist()} for n in nomes]
    # Confere se a entrada da primeira camada tem o número de colunas esperado (1.914).
    assert len(camadas[0]['W'][0]) == metadados['n_features'], 'dimensão de entrada divergente'
    # Grava o módulo de pesos com versão, ativação e camadas.
    gravar_modulo(DIR_ARTEFATOS / 'pesos.mjs', {'versao': versao, 'ativacao': 'relu', 'camadas': camadas})
    # Devolve a versão para os metadados.
    return versao


def exportar_preprocesso(preprocessador):
    """Extrai do ColumnTransformer ajustado tudo o que a transformação em JS precisa."""
    # Obtém o pipeline categórico ajustado (imputação 'missing' + OneHotEncoder).
    pipe_cat = preprocessador.named_transformers_['cat']
    # Obtém o pipeline numérico ajustado (imputação pela mediana + StandardScaler).
    pipe_num = preprocessador.named_transformers_['num']
    # Monta o dicionário coluna → categorias conhecidas, na ordem das colunas do one-hot.
    categorias = {col: [str(c) for c in cats] for col, cats in zip(CATEGORICAS_NOMINAIS, pipe_cat.named_steps['encoder'].categories_)}
    # Monta a lista de parâmetros numéricos (mediana, média e escala) por coluna derivada.
    numericas = [{'nome': col,
                  # Mediana do treino, usada quando o valor falta.
                  'mediana': float(med),
                  # Média do treino, subtraída no StandardScaler.
                  'media': float(mu),
                  # Desvio do treino, divisor no StandardScaler.
                  'escala': float(sd)}
                 # Percorre as três colunas numéricas com seus parâmetros ajustados.
                 for col, med, mu, sd in zip(NUMERICAS, pipe_num.named_steps['imputer'].statistics_,
                                             # Parâmetros do scaler na mesma ordem das colunas.
                                             pipe_num.named_steps['scaler'].mean_, pipe_num.named_steps['scaler'].scale_)]
    # Grava o módulo com a ordem das colunas (categóricas primeiro, depois numéricas, como no ColumnTransformer).
    gravar_modulo(DIR_ARTEFATOS / 'preprocesso.mjs', {'categoricas': CATEGORICAS_NOMINAIS, 'categorias': categorias,
                                                      # Valor usado pelo imputador para categorias ausentes.
                                                      'valor_ausente': 'missing', 'numericas': numericas})
    # Devolve as categorias para validar as opções do formulário.
    return categorias


def exportar_metadados(metadados, versao):
    """Grava os metadados públicos do modelo (sem caminhos locais)."""
    # Seleciona apenas os campos úteis à API e à interface.
    publico = {k: metadados[k] for k in ('architecture', 'hidden_sizes', 'n_features', 'n_parameters', 'activation',
                                         # Continua a lista de campos copiados de metadata.json.
                                         'threshold', 'best_epoch', 'selection_rule', 'val_f1_seeds_mean',
                                         # Métricas de confirmação no teste 2025.
                                         'val_f1_seeds_std', 'test_2025_accuracy', 'test_2025_f1_macro')}
    # Acrescenta a versão do modelo derivada do hash dos pesos.
    publico['versao_modelo'] = versao
    # Acrescenta a versão do contrato da API.
    publico['versao_contrato'] = '1.0'
    # Acrescenta a data da exportação em UTC.
    publico['exportado_em'] = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    # Grava o módulo de metadados.
    gravar_modulo(DIR_ARTEFATOS / 'metadados.mjs', publico)


def exportar_opcoes_e_casos(preprocessador, categorias):
    """Gera as listas do formulário a partir do treino 2025 e casos de paridade da validação."""
    # Carrega os microdados de 2025 já filtrados (São Paulo, roubo/furto).
    df_2025 = carregar_e_preparar(ano=2025)
    # Consolida por boletim e reproduz a mesma divisão do treino (semente fixa).
    treino, validacao, _ = preparar_split_2025(df_2025)
    # Conta os tipos de local do treino, do mais ao menos frequente.
    tipos = treino['DESCR_TIPOLOCAL'].dropna().value_counts().index.tolist()
    # Para cada tipo, lista os subtipos mais frequentes no treino.
    subtipos = {t: treino.loc[treino['DESCR_TIPOLOCAL'] == t, 'DESCR_SUBTIPOLOCAL'].dropna().value_counts().index[:MAX_SUBTIPOS].tolist()
                # Percorre todos os tipos de local encontrados.
                for t in tipos}
    # Lista os bairros mais frequentes do treino (texto exato do encoder), em ordem alfabética para facilitar a busca.
    bairros = sorted(treino['BAIRRO'].dropna().astype(str).value_counts().index[:MAX_BAIRROS].tolist(), key=str.strip)
    # Lista os períodos conhecidos pelo encoder, exceto o marcador interno de ausência.
    periodos = [p for p in categorias['DESCR_PERIODO'] if p != 'missing']
    # Monta o objeto de opções (somente rótulos; nenhum dado pessoal ou identificador).
    opcoes = {'tipos_local': tipos, 'subtipos_por_tipo': subtipos, 'bairros': bairros, 'periodos': periodos}
    # Cria a pasta de dados públicos se necessário.
    DIR_DADOS_PUBLICOS.mkdir(parents=True, exist_ok=True)
    # Grava o JSON de opções lido pelo formulário.
    (DIR_DADOS_PUBLICOS / 'opcoes.json').write_text(json.dumps(opcoes, ensure_ascii=False, indent=1), encoding='utf-8')
    # Informa o arquivo gravado.
    print(f'  gravado netlify/public/data/opcoes.json ({len(tipos)} tipos, {len(bairros)} bairros)')
    # Sorteia casos da validação (nunca do teste) para conferir a paridade entre Python e JavaScript.
    amostra = validacao.sample(n=N_CASOS_TESTE, random_state=0).reset_index(drop=True)
    # Transforma a amostra com o pré-processador ajustado (mesmo caminho do treino).
    X, _, _ = preparar_features(amostra, preprocessor=preprocessador, fit=False)
    # Devolve a amostra e a matriz transformada.
    return amostra, X


def calcular_referencia(metadados, X):
    """Calcula com PyTorch as probabilidades de roubo de referência para os casos de paridade."""
    # Carrega o state_dict original (chaves layers.N e output_layer, como no notebook).
    estado = torch.load(DIR_MODELO / 'mlp_model.pth', map_location='cpu', weights_only=True)
    # Converte a matriz transformada em tensor float32 (mesma precisão do treino).
    h = torch.tensor(X, dtype=torch.float32)
    # Percorre as camadas ocultas na ordem do forward (o dropout não atua na inferência).
    for i in range(len(metadados['hidden_sizes'])):
        # Aplica a camada linear (x·Wᵀ + b) seguida da ReLU.
        h = torch.relu(h @ estado[f'layers.{i}.weight'].T + estado[f'layers.{i}.bias'])
    # Aplica a camada de saída, que produz o logit da classe roubo.
    logit = h @ estado['output_layer.weight'].T + estado['output_layer.bias']
    # Converte o logit em probabilidade de roubo com a sigmoide e devolve um vetor numpy.
    return torch.sigmoid(logit).squeeze(1).numpy()


def gravar_casos(amostra, probs):
    """Grava os casos de paridade no formato de entrada da API, com a probabilidade esperada."""
    # Inicializa a lista de casos.
    casos = []
    # Percorre cada boletim sorteado e sua probabilidade de referência.
    for (_, linha), p in zip(amostra.iterrows(), probs):
        # Converte a data da ocorrência para texto AAAA-MM-DD (ou vazio, se ausente).
        data = '' if pd_isna(linha['DATA_OCORRENCIA_BO']) else str(linha['DATA_OCORRENCIA_BO'])[:10]
        # Converte a hora para texto HH:MM (ou vazio, se ausente/mal formatada).
        hora = str(linha['HORA_OCORRENCIA'])[:5] if ':' in str(linha['HORA_OCORRENCIA']) else ''
        # Monta o caso com os campos brutos do formulário e a probabilidade esperada.
        casos.append({'entrada': {'tipo_local': texto_ou_vazio(linha['DESCR_TIPOLOCAL']),
                                  # Subtipo do local do caso.
                                  'subtipo_local': texto_ou_vazio(linha['DESCR_SUBTIPOLOCAL']),
                                  # Bairro do caso.
                                  'bairro': texto_ou_vazio(linha['BAIRRO']),
                                  # Período do dia do caso.
                                  'periodo': texto_ou_vazio(linha['DESCR_PERIODO']),
                                  # Data e hora do caso.
                                  'data': data, 'hora': hora},
                      # Probabilidade de roubo calculada pelo PyTorch.
                      'prob_roubo_esperada': round(float(p), 6)})
    # Grava os casos em JSON (fora da pasta pública: não é servido ao navegador).
    (DIR_ARTEFATOS / 'casos_teste.json').write_text(json.dumps(casos, ensure_ascii=False, indent=1), encoding='utf-8')
    # Informa quantos casos foram gravados.
    print(f'  gravado netlify/functions/artefatos/casos_teste.json ({len(casos)} casos da validação)')


def pd_isna(valor):
    """Indica se `valor` é ausente (None, NaN ou NaT)."""
    # Importa pandas localmente para usar a mesma regra de ausência do treino.
    import pandas as pd
    # Devolve True quando o valor é ausente.
    return bool(pd.isna(valor))


def texto_ou_vazio(valor):
    """Converte `valor` em texto, ou em '' quando ausente (a API trata '' como ausente)."""
    # Devolve texto vazio para ausência e o próprio texto nos demais casos.
    return '' if pd_isna(valor) else str(valor)


def main():
    """Executa todas as exportações em sequência."""
    # Informa o início da exportação.
    print('Exportando artefatos para netlify/ ...')
    # Lê os metadados do modelo escolhido.
    metadados = json.loads((DIR_MODELO / 'metadata.json').read_text(encoding='utf-8'))
    # Carrega o pré-processador ajustado no treino.
    preprocessador = joblib.load(DIR_MODELO / 'preprocessor.pkl')
    # Exporta os pesos e obtém a versão do modelo.
    versao = exportar_pesos(metadados)
    # Exporta os parâmetros do pré-processamento.
    categorias = exportar_preprocesso(preprocessador)
    # Exporta os metadados públicos.
    exportar_metadados(metadados, versao)
    # Exporta as opções do formulário e obtém os casos de paridade.
    amostra, X = exportar_opcoes_e_casos(preprocessador, categorias)
    # Calcula as probabilidades de referência com PyTorch.
    probs = calcular_referencia(metadados, X)
    # Grava os casos de paridade.
    gravar_casos(amostra, probs)
    # Informa o fim da exportação.
    print('Exportação concluída.')


# Executa a exportação somente quando o arquivo é chamado como script.
if __name__ == '__main__':
    # Chama a função principal.
    main()
