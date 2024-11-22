import os
import shutil
import sys
from typing import Union

import numpy as np
import wfdb
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from tqdm import tqdm
from wfdb import processing

from src.config import (
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    TEST_DATA_DIR,
    TRAIN_DATA_DIR,
    VAL_DATA_DIR,
)


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
        logger.debug(f"{dataset_name} already exists in {dataset_dir}.")


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
        if "(AFIB" in annotation.aux_note[i]:
            j = i + 1
            while j < len(annotation.sample):
                if "(AFIB" not in annotation.aux_note[j]:
                    ann_vec[annotation.sample[i] : annotation.sample[j]] = 1
                    i = j
                    break
                j += 1
        i += 1

    return ann_vec


def split_data(data: np.ndarray, chunk_size: int) -> np.ndarray:
    """
    Split ECG data into chunks.

    :param data: ECG data.
    :param chunk_size: Number of samples per chunk.
    :return: Chunks of ECG data.
    """
    if not isinstance(data, np.ndarray):
        raise ValueError("data must be a numpy array")
    if not isinstance(chunk_size, int):
        raise ValueError("chunk_size must be an integer")

    num_samples = data.shape[0]
    if num_samples % chunk_size != 0:
        padding = chunk_size - (num_samples % chunk_size)
        logger.debug(f"Applying padding of size {padding} to the data.")
        if data.ndim == 1:
            data = np.pad(data, (0, padding), "constant")
        else:
            data = np.pad(data, ((0, padding), (0, 0)), "constant")

    num_chunks = num_samples // chunk_size
    chunks = [data[i * chunk_size : (i + 1) * chunk_size] for i in range(num_chunks)]

    return np.array(chunks)


def preprocess_record(
    record_name: str, dataset_dir: str, target_fs: int, interim_data_dir: str
) -> None:
    """
    Process a single ECG record.

    :param record_name: Name of the record.
    :param dataset_dir: Directory of the dataset.
    :param target_fs: Target sampling frequency.
    :param interim_data_dir: Directory to save interim data.
    """
    logger.debug(f"Loading {record_name} from {dataset_dir}...")
    try:
        record = wfdb.rdrecord(os.path.join(dataset_dir, record_name))
        annotation = wfdb.rdann(os.path.join(dataset_dir, record_name), "atr")
    except ValueError:
        return

    logger.debug(f"Resampling {record_name} to {target_fs} Hz...")
    resampled_x, resampled_ann = processing.resample_multichan(
        record.p_signal, annotation, record.fs, target_fs
    )
    vector_ann = aux2vec(resampled_ann, target_fs, resampled_x.shape[0])

    logger.debug(f"Saving {record_name} to {interim_data_dir}...")
    os.makedirs(interim_data_dir, exist_ok=True)

    np.savez(
        os.path.join(interim_data_dir, f"{record_name}.npz"),
        x=resampled_x,
        y=vector_ann,
    )
    logger.debug(f"{record_name} saved to {os.path.join(interim_data_dir, record_name + '.npz')}.")


def preprocess_dataset(dataset_name: str, target_fs: int, verbosity: str) -> None:
    """
    Process the entire dataset.

    :param dataset_name: Name of the dataset.
    :param target_fs: Target sampling frequency.
    :param verbosity: Verbosity level for logging.
    """
    logger.remove()
    logger.add(sys.stderr, level=verbosity.upper())

    dataset_dir = os.path.join(RAW_DATA_DIR, dataset_name)
    download_wfdb_dataset(dataset_name, dataset_dir)

    files = os.listdir(dataset_dir)
    files = [file.split(".")[0] for file in files if file.endswith(".hea")]

    logger.info(f"Preprocessing {len(files)} ECG records from {dataset_name}...")
    use_tqdm = verbosity.upper() != "DEBUG"
    file_iterator = (
        tqdm(files, desc=f"Preprocessing files from {dataset_name}") if use_tqdm else files
    )

    for file in file_iterator:
        preprocess_record(
            file, dataset_dir, target_fs, os.path.join(INTERIM_DATA_DIR, dataset_name)
        )

    logger.info(
        f"All preprocessed ECG data from {dataset_name} saved to {os.path.join(INTERIM_DATA_DIR, dataset_name)}"
    )


def save_chunks(chunks, info, save_dir, use_tqdm: bool) -> None:
    """
    Save chunks of data to the specified directory.

    :param chunks: List of data chunks.
    :param info: List of information tuples (dataset_name, file_name, channel, chunk_idx).
    :param save_dir: Directory to save the chunks.
    :param use_tqdm: Whether to use tqdm progress bar.
    """
    iterator = (
        tqdm(zip(chunks, info), desc=f"Saving data to {save_dir}", total=len(chunks), unit="chunk")
        if use_tqdm
        else zip(chunks, info)
    )
    for (chunk_x, chunk_y), (dataset_name, file_name, channel, chunk_idx) in iterator:
        np.savez(
            os.path.join(
                save_dir, f"{dataset_name}_{file_name}_chunk{chunk_idx}_channel{channel}.npz"
            ),
            x=normalize(chunk_x.reshape(-1, 1)).squeeze(),
            y=normalize(chunk_y.reshape(-1, 1)).squeeze(),
        )


def split_chunks(
    all_chunks: list, file_info: list, chunk_size: int, test_size: float, val_size: float
) -> tuple:
    """
    Split proportionally chunks to sets.

    :param all_chunks: list of all chunks from dataset
    :param file_info: list of file information for all chunks from dataset
    :param chunk_size: Number of samples per chunk.
    :param test_size: Proportion of the dataset to include in the test split.
    :param val_size: Proportion of the dataset to include in the validation split.
    """
    combined = list(zip(all_chunks, file_info))
    with_afib = [chunk for chunk in combined if np.sum(chunk[0][1]) > chunk_size * 0.1]
    without_afib = [chunk for chunk in combined if np.sum(chunk[0][1]) <= chunk_size * 0.1]
    logger.info(
        f"{len(with_afib)/len(all_chunks) * 100:.2f} % chunks contains atrial fibrillation"
    )

    def split_group(group):
        chunks, file_info = zip(*group)
        train_chunks, temp_chunks, train_info, temp_info = train_test_split(
            chunks, file_info, test_size=test_size + val_size, random_state=42
        )
        val_chunks, test_chunks, val_info, test_info = train_test_split(
            temp_chunks, temp_info, test_size=test_size / (test_size + val_size), random_state=42
        )
        return train_chunks, train_info, val_chunks, test_chunks, val_info, test_info

    (
        train_chunks_afib,
        train_info_afib,
        val_chunks_afib,
        test_chunks_afib,
        val_info_afib,
        test_info_afib,
    ) = split_group(with_afib)
    (
        train_chunks_NOafib,
        train_info_NOafib,
        val_chunks_NOafib,
        test_chunks_NOafib,
        val_info_NOafib,
        test_info_NOafib,
    ) = split_group(without_afib)

    train_chunks = train_chunks_afib + train_chunks_NOafib
    val_chunks = val_chunks_afib + val_chunks_NOafib
    test_chunks = test_chunks_afib + test_chunks_NOafib

    train_info = train_info_afib + train_info_NOafib
    val_info = val_info_afib + val_info_NOafib
    test_info = test_info_afib + test_info_NOafib

    def shuffle_together(chunks, log_info):
        combined = list(zip(chunks, log_info))
        np.random.shuffle(combined)
        shuffled_chunks, shuffled_log_info = zip(*combined)
        return list(shuffled_chunks), list(shuffled_log_info)

    train_chunks, train_info = shuffle_together(train_chunks, train_info)
    val_chunks, val_info = shuffle_together(val_chunks, val_info)
    test_chunks, test_info = shuffle_together(test_chunks, test_info)

    return train_chunks, train_info, val_chunks, val_info, test_chunks, test_info


def process_dataset(chunk_size: int, test_size: float, val_size: float, verbosity: str) -> None:
    """
    Process the preprocessed dataset.

    :param chunk_size: Number of samples per chunk.
    :param test_size: Proportion of the dataset to include in the test split.
    :param val_size: Proportion of the dataset to include in the validation split.
    :param verbosity: Verbosity level for logging.
    """
    logger.remove()
    logger.add(sys.stderr, level=verbosity.upper())

    processed_data_dir = PROCESSED_DATA_DIR
    train_dir = TRAIN_DATA_DIR
    val_dir = VAL_DATA_DIR
    test_dir = TEST_DATA_DIR
    os.makedirs(INTERIM_DATA_DIR, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    all_files = []
    for dataset_name in os.listdir(INTERIM_DATA_DIR):
        interim_data_dir = os.path.join(INTERIM_DATA_DIR, dataset_name)
        if os.path.isdir(interim_data_dir):
            files = [
                (dataset_name, os.path.join(interim_data_dir, file))
                for file in os.listdir(interim_data_dir)
                if file.endswith(".npz")
            ]
            all_files.extend(files)

    logger.info(f"Processing {len(all_files)} preprocessed ECG records from all datasets...")
    use_tqdm = verbosity.upper() != "DEBUG"
    file_iterator = (
        tqdm(all_files, desc="Processing files", unit="file") if use_tqdm else all_files
    )

    all_chunks = []
    file_info = []

    for dataset_name, file in file_iterator:
        logger.debug(f"Processing file {file} from dataset {dataset_name}...")
        data = np.load(file)
        x = data["x"]
        y = data["y"]

        for channel in range(x.shape[1]):
            chunks_x = split_data(x[:, channel], chunk_size)
            chunks_y = split_data(y, chunk_size)

            all_chunks.extend(zip(chunks_x, chunks_y))
            file_info.extend(
                [
                    (dataset_name, os.path.basename(file).split(".")[0], channel, i)
                    for i in range(len(chunks_x))
                ]
            )

    logger.info(
        f"Splitting data into training, validation, and test sets with test size {test_size} and validation size {val_size}..."
    )

    train_chunks, train_info, val_chunks, val_info, test_chunks, test_info = split_chunks(
        all_chunks, file_info, chunk_size, test_size, val_size
    )

    logger.info(f"Saving training data to {train_dir}...")
    save_chunks(train_chunks, train_info, train_dir, use_tqdm)

    logger.info(f"Saving validation data to {val_dir}...")
    save_chunks(val_chunks, val_info, val_dir, use_tqdm)

    logger.info(f"Saving test data to {test_dir}...")
    save_chunks(test_chunks, test_info, test_dir, use_tqdm)

    logger.info(f"All processed ECG data saved to {processed_data_dir}")


def clear_data(dataset_name: Union[str, None], data_type: str) -> None:
    """
    Clear interim or processed data.

    :param dataset_name: Name of the dataset or None to clear all datasets.
    :param data_type: Type of data to clear ('interim', 'processed', or 'both').
    """
    if data_type not in ["interim", "processed", "both"]:
        raise ValueError("data_type must be 'interim', 'processed', or 'both'")

    dirs_to_clear = []
    if dataset_name:
        if data_type in ["interim", "both"]:
            dirs_to_clear.append(os.path.join(INTERIM_DATA_DIR, dataset_name))
        if data_type in ["processed", "both"]:
            dirs_to_clear.append(TRAIN_DATA_DIR)
            dirs_to_clear.append(VAL_DATA_DIR)
            dirs_to_clear.append(TEST_DATA_DIR)
    else:
        if data_type in ["interim", "both"]:
            dirs_to_clear.append(INTERIM_DATA_DIR)
        if data_type in ["processed", "both"]:
            dirs_to_clear.append(TRAIN_DATA_DIR)
            dirs_to_clear.append(VAL_DATA_DIR)
            dirs_to_clear.append(TEST_DATA_DIR)

    for dir_to_clear in dirs_to_clear:
        if os.path.exists(dir_to_clear):
            for file in os.listdir(dir_to_clear):
                file_path = os.path.join(dir_to_clear, file)
                if os.path.isfile(file_path) and file != ".gitkeep":
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            if not os.listdir(dir_to_clear):
                os.rmdir(dir_to_clear)
            logger.info(f"Cleared {data_type} data in {dir_to_clear}")
        else:
            logger.info(f"No {data_type} data found in {dir_to_clear}")
