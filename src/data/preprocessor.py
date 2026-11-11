"""Pré-processamento: tratamento de ausências, encoding categórico, scaling numérico.

O fit é feito APENAS no conjunto de treino; transform é aplicado em validação e teste.
"""

# Importa ferramentas de pré-processamento do scikit-learn.
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np

from src.utils.config import RANDOM_SEED

COLUNAS_FEATURES = [
    "DESCR_TIPOLOCAL",      # Categórica: tipo de local (ex: VIA PUBLICA, RESIDENCIA)
    "DESCR_SUBTIPOLOCAL",   # Categórica: subtipo (ex: CALÇADA, INTERIOR DE RESIDENCIA)
    "BAIRRO",               # Categórica: bairro (alta cardinalidade → OneHot ou TargetEncoder)
    "DESCR_PERIODO",        # Categórica: MANHA, TARDE, NOITE, MADRUGADA
    "HORA_OCORRENCIA",      # String HH:MM:SS → extrair hora inteira (numérica 0-23)
    "DATA_OCORRENCIA_BO",   # Datetime → extrair dia_da_semana (0-6) e mes (1-12)
]

COLUNAS_DERIVADAS = ["dia_semana", "mes", "hora"]

CATEGORICAS_NOMINAIS = ["DESCR_TIPOLOCAL", "DESCR_SUBTIPOLOCAL", "BAIRRO", "DESCR_PERIODO"]

NUMERICAS = ["dia_semana", "mes", "hora"]

def extrair_features_temporais(df: pd.DataFrame) -> pd.DataFrame:
    """Deriva colunas numéricas a partir de data e hora da ocorrência.

    Args:
        df: DataFrame contendo DATA_OCORRENCIA_BO (datetime) e HORA_OCORRENCIA (string).

    Returns:
        DataFrame com colunas adicionais: dia_semana (0-6), mes (1-12), hora (0-23).
    """
    df = df.copy()
    # Garante que DATA_OCORRENCIA_BO seja datetime.
    df["DATA_OCORRENCIA_BO"] = pd.to_datetime(df["DATA_OCORRENCIA_BO"], errors="coerce")
    # Extrai dia da semana (0=segunda ... 6=domingo).
    df["dia_semana"] = df["DATA_OCORRENCIA_BO"].dt.dayofweek
    # Extrai mês (1-12).
    df["mes"] = df["DATA_OCORRENCIA_BO"].dt.month

    # Converte HORA_OCORRENCIA (string "HH:MM:SS") para hora inteira.
    # Valores nulos ou mal formatados viram NaN.
    def parse_hora(valor):
        if pd.isna(valor):
            return np.nan
        try:
            return int(str(valor).split(":") [0])

        except(ValueError, IndexError):
            return np.nan

    df["hora"] = df["HORA_OCORRENCIA"].apply(parse_hora)
    return df

def construir_preprocessor() -> ColumnTransformer:
    """Constrói o ColumnTransformer com pipelines para categóricas e numéricas.

    - Categóricas: imputa com 'missing' + OneHotEncoder (handle_unknown='ignore').
    - Numéricas: imputa com mediana + StandardScaler.

    Returns:
        ColumnTransformer ajustável (fit) e transformável.
    """
    # Pipeline para variáveis categóricas nominais.
    pipe_cat = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    # Pipeline para variáveis numéricas.
    pipe_num = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # Combina os pipelines por tipo de coluna.
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", pipe_cat, CATEGORICAS_NOMINAIS),
            ("num", pipe_num, NUMERICAS),
        ],
        remainder="drop",  # Descarta colunas não listadas (ex: NUM_BO, RUBRICA, etc.)
        verbose_feature_names_out=False,
    )
    return preprocessor

def preparar_features(df: pd.DataFrame, preprocessor: ColumnTransformer | None = None,
                      fit: bool = False) -> tuple[pd.DataFrame, ColumnTransformer]:
    """Aplica derivação temporal e transforma features com o preprocessor.

    Args:
        df: DataFrame bruto (após loader: filtrado SP, com target).
        preprocessor: ColumnTransformer já ajustado (se None, cria novo).
        fit: Se True, ajusta o preprocessor nos dados (apenas no treino!).

    Returns:
        Tupla (X_transformado, preprocessor_ajustado).
        X_transformado é array numpy denso (n_samples, n_features).
    """
    # 1. Deriva variáveis temporais
    df = extrair_features_temporais(df)

    # 2. Seleciona apenas as colunas de features + target para segurança
    colunas_necessarias = CATEGORICAS_NOMINAIS + NUMERICAS + ["target"]
    df = df[colunas_necessarias].copy()

    # 3. Separa features (X) e target (y).
    X = df.drop(columns=["target"])
    y = df["target"].values

    # 4. Instancia ou reusa o preprocessor.
    if preprocessor is None:
        preprocessor = construir_preprocessor()

    # 5. Fit (apenas treino) ou transform (validação/teste).
    if fit:
        X_transformado = preprocessor.fit_transform(X)
    else:
        X_transformado = preprocessor.transform(X)

    return X_transformado, y, preprocessor