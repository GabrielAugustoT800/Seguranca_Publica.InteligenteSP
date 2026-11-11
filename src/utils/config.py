"""Configurações centrais do projeto (paths, sementes, hiperparâmetros)."""

from pathlib import Path

# Diretórios raiz
ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
BEST_MODEL_DIR = MODELS_DIR / "best"
EXPERIMENTS_DIR = MODELS_DIR / "experiments"
ARCHIVE_DIR = MODELS_DIR / "archive"
REPORTS_DIR = ROOT / "reports"
SLIDES_DIR = REPORTS_DIR / "slides"
IMAGES_DIR = REPORTS_DIR / "images"

# Semente fixa para reprodutibilidade
RANDOM_SEED = 42

# Particionamento dos dados
TRAIN_SIZE = 0.70
VAL_SIZE = 0.15
TEST_SIZE = 0.15

# Hiperparâmetros base da MLP
LEARNING_RATE = 1e-3
EPOCHS = 50
BATCH_SIZE = 32

# Arquiteturas a serem testadas (exemplos)
ARCHITECTURES = {
    "mlp_32": [32],
    "mlp_64_32": [64, 32],
    "mlp_128_64_32": [128, 64, 32],
}

# Target
TARGET_COLUMN = "tipo_ocorrencia"  # ajustar conforme a base escolhida
LABELS = ["furto", "roubo"]
