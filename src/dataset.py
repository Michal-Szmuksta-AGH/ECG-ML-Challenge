import os
import numpy as np
import wfdb
from loguru import logger
from typing import Union


def download_wfdb_dataset(dataset_name: str, dataset_dir: str) -> None:
    """
    Load WFDB dataset.

    :param dataset_name: Name of the dataset.
    :param dataset_dir: Directory to save the dataset.
    """

    if not isinstance(dataset_name, str):
        raise ValueError("dataset_name must be a string")
    if not isinstance(dataset_dir, str):
        raise ValueError("dataset_dir must be a string")

    if not os.path.exists(dataset_dir) or len(os.listdir(dataset_dir)) == 0:
        logger.info(f"{dataset_name} not found in {dataset_dir}.")
        logger.info(f"Downloading {dataset_name} database into {dataset_dir}...")
        wfdb.dl_database(dataset_name, dl_dir=dataset_dir)
        logger.info(f"{dataset_name} database downloaded.")
    else:
        logger.info(f"{dataset_name} already exists in {dataset_dir}.")


def ann2vec(annotation: wfdb.Annotation, fs: int, samples: Union[None, int] = None) -> np.ndarray:
    """
    Convert WFDB annotation to vector.

    :param annotation: WFDB annotation object.
    :param fs: Sampling frequency.
    :param samples: Number of output samples.
    :return: Annotation vector.
    """
    if not isinstance(annotation, wfdb.Annotation):
        raise ValueError("annotation must be a wfdb.Annotation object")
    if not isinstance(fs, int):
        raise ValueError("fs must be an integer")

    if samples is None:
        samples = annotation.sample[-1]

    ann_vec = np.zeros(samples, dtype=int)
    for i in range(len(annotation.sample)):
        if annotation.symbol[i] == "N":
            continue
        start_sample = annotation.sample[i - 1] // 2 if i > 0 else 0
        end_sample = annotation.sample[i + 1] // 2 if i < len(annotation.sample) - 1 else samples

        ann_vec[start_sample:end_sample] = 1

    return ann_vec
