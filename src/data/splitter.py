"""Consolidação de boletins e separação estratificada train/val/test.

- Um mesmo NUM_BO pode ter múltiplas linhas (itens múltiplos).
- Consolida: uma linha por BO, target = max(target) (se qualquer item for roubo, BO = roubo).
- Split: 70% treino, 15% validação, 15% teste, estratificado pelo target.
- 2025 → treino + validação; 2026 → teste final (não misturar anos).
"""

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

from src.utils.config import RANDOM_SEED, TRAIN_SIZE, VAL_SIZE, TEST_SIZE

def consolidar_boletins(df: pd.DataFrame) -> pd.DataFrame:
    """Consolida múltiplas linhas do mesmo BO em uma única linha.

    Regra: se o boletim tem ao menos um item de ROUBO (target=1), o BO inteiro vira ROUBO.
    Para features categóricas, usa a primeira ocorrência (mode). Para numéricas, mediana.

    Args:
        df: DataFrame com coluna NUM_BO (chave) e target (0/1) já definido.

    Returns:
        DataFrame com uma linha por NUM_BO, target consolidado, features agregadas.
    """
    # Define como agregar cada coluna na consolidação.
    agg_funcs = {
        "target": "max",                    # 1 se qualquer item for roubo
        "DESCR_TIPOLOCAL": lambda x: x.mode().iloc[0] if not x.mode().empty else "missing",
        "DESCR_SUBTIPOLOCAL": lambda x: x.mode().iloc[0] if not x.mode().empty else "missing",
        "BAIRRO": lambda x: x.mode().iloc[0] if not x.mode().empty else "missing",
        "DESCR_PERIODO": lambda x: x.mode().iloc[0] if not x.mode().empty else "missing",
        "HORA_OCORRENCIA": lambda x: x.mode().iloc[0] if not x.mode().empty else "missing",
        "DATA_OCORRENCIA_BO": "first",      # Data da primeira ocorrência do BO
    }

    # Agrupa por NUM_BO e aplica as funções de agregação.
    df_consolidado = df.groupby("NUM_BO", as_index=False).agg(agg_funcs)
    return df_consolidado

def split_estratificado(df: pd.DataFrame, 
                        train_size: float = TRAIN_SIZE,
                        val_size: float = VAL_SIZE,
                        test_size: float = TEST_SIZE,
                        seed: int = RANDOM_SEED) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Divide o DataFrame consolidado em treino, validação e teste estratificados.

    Mantém a proporção de classes (target) em cada conjunto.
    Garante que um mesmo NUM_BO não apareça em mais de um conjunto.

    Args:
        df: DataFrame consolidado (uma linha por BO).
        train_size: Proporção para treino (default 0.70).
        val_size: Proporção para validação (default 0.15).
        test_size: Proporção para teste (default 0.15).
        seed: Semente para reprodutibilidade.

    Returns:
        Tupla (df_train, df_val, df_test) — DataFrames com os índices originais preservados.
    """
    # Valida se as proporções somam 1.
    total = train_size + val_size + test_size
    if not np.isclose(total, 1.0):
        raise ValueError(f"Proporções devem somar 1.0, somam {total}")

    # Primeira divisão: separa teste (15%) do resto (85%).
    df_temp, df_test = train_test_split(
        df,
        test_size=test_size,
        stratify=df["target"],
        random_state=seed,
    )

    # Segunda divisão: do temp (85%), separa validação (15/85 ≈ 17.65%) e treino (70/85 ≈ 82.35%).
    val_relativo = val_size / (train_size + val_size)
    df_train, df_val = train_test_split(
        df_temp,
        test_size=val_relativo,
        stratify=df_temp["target"],
        random_state=seed,
    )

    # Reseta índices para evitar índices duplicados downstream.
    return df_train.reset_index(drop=True), df_val.reset_index(drop=True), df_test.reset_index(drop=True)

def preparar_split_2025(df_2025: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Pipeline completa para 2025: consolida BOs + split estratificado 70/15/15.

    Args:
        df_2025: DataFrame bruto de 2025 (já filtrado SP, com target).

    Returns:
        Tupla (train, val, test) — DataFrames consolidados e particionados.
    """
    # Consolida múltiplas linhas por BO.
    df_consolidado = consolidar_boletins(df_2025)
    # Divide estratificamente.
    return split_estratificado(df_consolidado)

def preparar_teste_2026(df_2026: pd.DataFrame) -> pd.DataFrame:
    """Prepara o conjunto de teste final a partir de 2026: apenas consolida BOs.

    Não faz split — o arquivo 2026 inteiro (após consolidação) vira o teste final.

    Args:
        df_2026: DataFrame bruto de 2026 (já filtrado SP, com target).

    Returns:
        DataFrame consolidado (uma linha por BO) para teste final.
    """
    return consolidar_boletins(df_2026)
