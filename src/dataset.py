import os
import sys
import wfdb
import typer
import numpy as np
from loguru import logger
from typing import Union
from wfdb import processing
from tqdm import tqdm

from .config import RAW_DATA_DIR, PROCESSED_DATA_DIR


app = typer.Typer()


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


def sym2vec(annotation: wfdb.Annotation, fs: int, samples: Union[None, int] = None) -> np.ndarray:
    """
    Convert WFDB annotation symbol to vector.

    :param annotation: WFDB annotation object.
    :param fs: Sampling frequency.
    :param samples: Number of output samples.
    :return: Annotation vector.
    """
    if not isinstance(annotation, wfdb.Annotation):
        raise ValueError("annotation must be a wfdb.Annotation object")
    if not isinstance(fs, int):
        raise ValueError("fs must be an integer")
    if samples is not None and not isinstance(samples, int):
        raise ValueError("samples must be an integer")

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

def aux2vec(annotation: wfdb.Annotation, fs: int, samples: Union[None, int] = None) -> np.ndarray:
    """
    Convert WFDB annotation aux note to vector.

    :param annotation: WFDB annotation object.
    :param fs: Sampling frequency.
    :param samples: Number of output samples.
    :return: Annotation vector.
    """
    if not isinstance(annotation, wfdb.Annotation):
        raise ValueError("annotation must be a wfdb.Annotation object")
    if not isinstance(fs, int):
        raise ValueError("fs must be an integer")
    if samples is not None and not isinstance(samples, int):
        raise ValueError("samples must be an integer")

    if samples is None:
        samples = annotation.sample[-1]

    ann_vec = np.zeros(samples, dtype=np.uint8)
    i = 0
    while i < len(annotation.sample):
        if annotation.aux_note[i] == "(AFIB":
            j = i + 1
            while j < len(annotation.sample):
                if annotation.aux_note[j] != "(AFIB":
                    ann_vec[annotation.sample[i]:annotation.sample[j]] = 1
                    i = j
                    break
                j += 1
        i += 1

    return ann_vec


def split_data(data: np.ndarray, fs: int, chunk_size: int) -> np.ndarray:
    """
    Split ECG data into chunks.

    :param data: ECG data.
    :param fs: Sampling frequency.
    :param chunk_size: Number of samples per chunk.
    :return: Chunks of ECG data.
    """
    if not isinstance(data, np.ndarray):
        raise ValueError("data must be a numpy array")
    if not isinstance(fs, int):
        raise ValueError("fs must be an integer")
    if not isinstance(chunk_size, int):
        raise ValueError("chunk_size must be an integer")

    num_samples = data.shape[0]
    num_chunks = num_samples // chunk_size

    chunks = [data[i * chunk_size : (i + 1) * chunk_size] for i in range(num_chunks)]

    return np.array(chunks)


@app.command()
def main(dataset_name: str, target_fs: int, verbosity: str = "INFO") -> None:
    """
    Load WFDB dataset.

    :param dataset_name: Name of the dataset.
    :param target_fs: Target sampling frequency.
    :param verbosity: Verbosity level for logging.
    """
    logger.remove()
    logger.add(sys.stderr, level=verbosity.upper())

    dataset_dir = os.path.join(RAW_DATA_DIR, dataset_name)
    download_wfdb_dataset(dataset_name, dataset_dir)

    # Load ECG data
    files = os.listdir(dataset_dir)
    files = [file.split(".")[0] for file in files if file.endswith(".hea")]
    logger.info(f"Processing {len(files)} ECG records...")

    use_tqdm = verbosity.upper() != "DEBUG"
    file_iterator = tqdm(files, desc="Processing files") if use_tqdm else files

    for file in file_iterator:
        record_name = file.split(".")[0]
        logger.debug(f"Loading {record_name}...")
        try:
            record = wfdb.rdrecord(os.path.join(dataset_dir, record_name))
            annotation = wfdb.rdann(os.path.join(dataset_dir, record_name), "atr")
        except ValueError:
            continue

        # Resample ECG data
        logger.debug(f"Resampling {record_name} to {target_fs} Hz...")
        resampled_x, resampled_ann = processing.resample_multichan(
            record.p_signal, annotation, record.fs, target_fs
        )
        vector_ann = aux2vec(resampled_ann, target_fs)

        # Save resampled ECG data
        logger.debug(f"Saving {record_name}...")
        os.makedirs(os.path.join(PROCESSED_DATA_DIR, dataset_name), exist_ok=True)
        np.savez(
            os.path.join(PROCESSED_DATA_DIR, dataset_name, f"{record_name}.npz"),
            x=resampled_x,
            y=vector_ann,
        )
        logger.debug(
            f"{record_name} in {os.path.join(PROCESSED_DATA_DIR, dataset_name, record_name + '.npz')} saved."
        )

    logger.info(f"All ECG data saved to {os.path.join(PROCESSED_DATA_DIR, dataset_name)}")
    return


if __name__ == "__main__":
    app()
