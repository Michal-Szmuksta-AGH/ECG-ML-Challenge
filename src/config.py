import os
from pathlib import Path

from loguru import logger

PROJ_ROOT = Path(__file__).resolve().parents[1]
# Zakomentowałem bo wkurwiał mnie ten komunikat wyświetlany absolutnie wszędzie
# logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

TRAIN_DATA_DIR = PROCESSED_DATA_DIR / "train"
TEST_DATA_DIR = PROCESSED_DATA_DIR / "test"
VAL_DATA_DIR = PROCESSED_DATA_DIR / "val"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

MINIMAL_RUNTIME_DIR = PROJ_ROOT / "minimal_runtime"

SRC_DIR = PROJ_ROOT / "src"

os.environ["WANDB_DIR"] = str(REPORTS_DIR)

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
