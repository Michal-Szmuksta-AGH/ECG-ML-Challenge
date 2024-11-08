import os
import wfdb
from loguru import logger

def download_wfdb_dataset(dataset_name: str, dataset_dir: str) -> None:
    """
    Load WFDB dataset.

    :param dataset_name: Name of the dataset.
    :param dataset_dir: Directory to save the dataset.
    """

    if not isinstance(dataset_name, str):
        raise ValueError('dataset_name must be a string')
    if not isinstance(dataset_dir, str):
        raise ValueError('dataset_dir must be a string')

    if not os.path.exists(dataset_dir) or len(os.listdir(dataset_dir)) == 0:
        logger.info(f'{dataset_name} not found in {dataset_dir}.')
        logger.info(f'Downloading {dataset_name} database into {dataset_dir}...')
        os.makedirs(dataset_dir, exist_ok=True)
        wfdb.dl_database(dataset_name, dl_dir=dataset_dir)
        logger.info(f'{dataset_name} database downloaded.')
    else:
        logger.info(f'{dataset_name} already exists in {dataset_dir}.')