import os

from src import config  # noqa: F401

os.makedirs(config.DATA_DIR, exist_ok=True)
os.makedirs(config.RAW_DATA_DIR, exist_ok=True)
os.makedirs(config.INTERIM_DATA_DIR, exist_ok=True)
os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(config.EXTERNAL_DATA_DIR, exist_ok=True)
os.makedirs(config.MODELS_DIR, exist_ok=True)
os.makedirs(config.REPORTS_DIR, exist_ok=True)
os.makedirs(config.FIGURES_DIR, exist_ok=True)
