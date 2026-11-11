"""Carregameento e validação dos dados de celulares subtraídos ds SSP-SP"""

from pathlib import Path
import pandas as pd
from src.utils.config import DATA_RAW, RANDOM_SEED

SHEET_2025 = "CELULAR_2025"
SHEET_2026 = "CELULAR_2026"

ARQUIVOS = {
    2025: DATA_RAW/"CelularesSubtraidos_2025.xlsx",
    2026: DATA_RAW/"CelularesSubtraidos_2026.xlsx"
}

ABAS = {
    2025: SHEET_2025,
    2026: SHEET_2026
}

COLUNAS_OBRIGATORIAS = [
    "RUBRICA",             # Variável alvo: natureza da ocorrência (roubo/furto)
    "NOME_MUNICIPIO",      # Município do fato (filtrar São Paulo capital)
    "DESCR_TIPOLOCAL",     # Tipo de local da ocorrência
    "DESCR_SUBTIPOLOCAL",  # Subtipo de local
    "BAIRRO",              # Bairro da ocorrência
    "DESCR_PERIODO",       # Período do dia (manhã/tarde/noite/madrugada)
    "HORA_OCORRENCIA",     # Hora da ocorrência (string HH:MM:SS)
    "DATA_OCORRENCIA_BO",  # Data da ocorrência (datetime)
    "NUM_BO",              # Número do boletim de ocorrência (chave de consolidação)
]

def carregar_ano(ano: int, linhas: int | None = None) -> pd.DataFrame:
    """Lê o arquivo Excel do ano especificado e retorna um DataFrame.

    Args:
        ano: 2025 (treino/validação) ou 2026 (teste final).
        linhas: Se informado, lê apenas as N primeiras linhas (para inspeção rápida).

    Returns:
        DataFrame com os dados brutos do ano solicitado.

    Raises:
        FileNotFoundError: Se o arquivo não existir em data/raw/.
        ValueError: Se o ano não for 2025 nem 2026.
    """
    if ano not in ARQUIVOS:
        raise ValueError(f"Ano {ano} não suportado. Use 2025 ou 2026.")

    caminho = ARQUIVOS[ano]
    if not caminho.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho}")

    df = pd.read_excel(caminho, sheet_name=ABAS[ano], nrows= linhas, engine = "openpyxl")
    return df

def validar_colunas(df: pd.DataFrame, ano: int) -> None:
    """Verifica se todas as colunas obrigatórias existem no DataFrame.

    Args:
        df: DataFrame carregado.
        ano: Ano dos dados (para mensagem de erro).

    Raises:
        ValueError: Se alguma coluna obrigatória estiver ausente.
    """
    faltando = [c for c in COLUNAS_OBRIGATORIAS if c not in df.columns]
    if faltando:
        raise ValueError(f"Colunas obrigatórias ausentes no dataset: {ano}: {faltando}")

def filtar_sao_paulo(df:pd.DataFrame) -> pd.DataFrame:
    """Filtra apenas ocorrências cujo município do fato seja São Paulo (capital).

    O campo NOME_MUNICIPIO usa 'S.PAULO' para a capital.
    Remove também linhas onde o município seja nulo.

    Args:
        df: DataFrame com coluna NOME_MUNICIPIO.

    Returns:
        DataFrame filtrado apenas para S.PAULO, cópia para evitar SettingWithCopyWarning.
    """
    mask = df["NOME_MUNICIPIO"] == "S.PAULO"
    return df.loc[mask].copy()

def definir_alvo(df:pd.DataFrame) -> pd.DataFrame:
    """Converte a coluna RUBRICA em alvo binário: 1 = Roubo, 0 = Furto.

    Exclui registros que não sejam nem roubo nem furto (ex: perda, extravio, outros).

    Args:
        df: DataFrame contendo a coluna RUBRICA.

    Returns:
        DataFrame filtrado (apenas roubo/furto) com nova coluna 'target' (0/1).
    """
    mapa = {
        "Furto (art. 155)": 0,
        "Roubo (art. 157)": 1
    }

    df_filtrado = df[df["RUBRICA"].isin(mapa.keys())].copy()
    df_filtrado["target"] = df_filtrado["RUBRICA"].map(mapa)
    return df_filtrado

def carregar_e_preparar(ano: int, linhas: int | None = None) -> pd.DataFrame:
    """Pipeline completa: carrega, valida, filtra São Paulo e define alvo.

    Args:
        ano: 2025 ou 2026.
        linhas: Limite de linhas para leitura rápida (None = completo).

    Returns:
        DataFrame pronto para pré-processamento (apenas SP, apenas roubo/furto, com target).
    """
    df = carregar_ano(ano, linhas = linhas)
    validar_colunas(df,ano)
    df = filtar_sao_paulo(df)
    df = definir_alvo(df)
    return df