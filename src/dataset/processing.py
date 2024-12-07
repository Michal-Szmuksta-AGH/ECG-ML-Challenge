import os
import shutil
import sys
from typing import Union

import numpy as np
import wfdb
from loguru import logger
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from wfdb import processing
from scipy.signal import firwin, lfilter
import pandas as pd

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


def fir_bandpass_filter(data, lowcut, highcut, fs, numtaps=101):
    """
    Apply a bandpass FIR filter to the data.

    :param data: Input data.
    :param lowcut: Low cutoff frequency.
    :param highcut: High cutoff frequency.
    :param fs: Sampling frequency.
    :param numtaps: Number of filter taps.
    :return: Filtered data.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    taps = firwin(numtaps, [low, high], pass_zero=False)
    y = lfilter(taps, 1.0, data, axis=0)
    return y


def preprocess_record(
    record_name: str, dataset_dir: str, target_fs: int, interim_data_dir: str, chunk_size: int
) -> None:
    """
    Process a single ECG record.

    :param record_name: Name of the record.
    :param dataset_dir: Directory of the dataset.
    :param target_fs: Target sampling frequency.
    :param interim_data_dir: Directory to save interim data.
    :param chunk_size: Number of samples per chunk.
    """
    logger.debug(f"Loading {record_name} from {dataset_dir}...")
    dataset_name = os.path.basename(dataset_dir)
    try:
        record = wfdb.rdrecord(os.path.join(dataset_dir, record_name))
        annotation = wfdb.rdann(os.path.join(dataset_dir, record_name), "atr")
    except:
        try:
            record = wfdb.rdrecord(os.path.join(dataset_dir, record_name))
            annotation = wfdb.rdann(os.path.join(dataset_dir, record_name), "qrs")
        except:
            return

    logger.debug(f"Resampling {record_name} to {target_fs} Hz...")
    resampled_x, resampled_ann = processing.resample_multichan(
        record.p_signal, annotation, record.fs, target_fs
    )
    if dataset_dir.endswith("af-termination-challenge"):
        vector_ann = np.ones(resampled_x.shape[0], dtype=np.uint8)
    else:
        vector_ann = aux2vec(resampled_ann, target_fs, resampled_x.shape[0])

    logger.debug(f"Filtering {record_name}...")
    filtered_x = fir_bandpass_filter(
        resampled_x, lowcut=0.5, highcut=30, fs=target_fs, numtaps=201
    )

    logger.debug(f"Splitting {record_name} into chunks of size {chunk_size}...")
    chunks_x = split_data(filtered_x, chunk_size)
    chunks_y = split_data(vector_ann, chunk_size)

    logger.debug(f"Scaling chunks of {record_name}...")
    for channel in range(filtered_x.shape[1]):
        scaler_x = MinMaxScaler(feature_range=(0, 1))
        for chunk in chunks_x:
            scaler_x.partial_fit(chunk[:, channel].reshape(-1, 1))
        for chunk in chunks_x:
            chunk[:, channel] = scaler_x.transform(chunk[:, channel].reshape(-1, 1)).squeeze()

    logger.debug(f"Saving chunks of {record_name} to {interim_data_dir}...")
    os.makedirs(interim_data_dir, exist_ok=True)

    for i, (chunk_x, chunk_y) in enumerate(zip(chunks_x, chunks_y)):
        for channel in range(chunk_x.shape[1]):
            np.savez(
                os.path.join(interim_data_dir, f"{dataset_name}#{record_name}#{channel}#{i}.npz"),
                x=chunk_x[:, channel],
                y=chunk_y,
            )
    logger.debug(f"Chunks of {record_name} saved to {interim_data_dir}.")


def preprocess_dataset(dataset_name: str, target_fs: int, chunk_size: int, verbosity: str) -> None:
    """
    Process the entire dataset.

    :param dataset_name: Name of the dataset.
    :param target_fs: Target sampling frequency.
    :param chunk_size: Number of samples per chunk.
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

    interim_data_dir = os.path.join(INTERIM_DATA_DIR, dataset_name)
    for file in file_iterator:
        preprocess_record(file, dataset_dir, target_fs, interim_data_dir, chunk_size)

    logger.info(f"All preprocessed ECG data from {dataset_name} saved to {interim_data_dir}")


def calculate_distances(
    train_count, val_count, test_count, group_size, total_current_samples, proportions
):
    predicted_train_dist = (train_count + group_size) / (total_current_samples + group_size)
    predicted_val_dist = (val_count + group_size) / (total_current_samples + group_size)
    predicted_test_dist = (test_count + group_size) / (total_current_samples + group_size)

    train_dist = np.linalg.norm(
        np.array(
            [
                predicted_train_dist,
                val_count / (total_current_samples + group_size),
                test_count / (total_current_samples + group_size),
            ]
        )
        - proportions
    )
    val_dist = np.linalg.norm(
        np.array(
            [
                train_count / (total_current_samples + group_size),
                predicted_val_dist,
                test_count / (total_current_samples + group_size),
            ]
        )
        - proportions
    )
    test_dist = np.linalg.norm(
        np.array(
            [
                train_count / (total_current_samples + group_size),
                val_count / (total_current_samples + group_size),
                predicted_test_dist,
            ]
        )
        - proportions
    )

    return train_dist, val_dist, test_dist


def iterative_group_balanced_split(df, stratify_col, group_col, proportions, random_state=42):
    """
    Split data into training, validation, and test sets while maintaining class and group balance.

    :param df: DataFrame with data.
    :param stratify_col: Column with class labels (e.g., 0 for non-AFIB, 1 for AFIB).
    :param group_col: Column identifying groups.
    :param proportions: Tuple with split proportions (train, val, test).
    :param random_state: Seed for reproducibility.
    :return: train_set, val_set, test_set (DataFrames).
    """
    assert sum(proportions) == 1, "Proportions must sum to 1."

    train_set = pd.DataFrame()
    val_set = pd.DataFrame()
    test_set = pd.DataFrame()

    unused_samples = {"train": [], "val": [], "test": []}
    excess_afib_samples = []

    train_count = val_count = test_count = 0

    group_sizes = df[group_col].value_counts().to_dict()
    groups = sorted(group_sizes.keys(), key=lambda x: group_sizes[x], reverse=True)

    # Phase 1: Iterating through groups
    for group in tqdm(groups, desc="Processing groups in phase 1"):
        group_data = df[df[group_col] == group]

        # Split group into classes
        non_afib_samples = group_data[group_data[stratify_col] == 0]
        afib_samples = group_data[group_data[stratify_col] == 1]

        # Balance the group
        if len(non_afib_samples) > len(afib_samples):
            # Excess non-AFIB samples
            excess_non_afib = non_afib_samples.sample(
                len(non_afib_samples) - len(afib_samples), random_state=random_state
            )
            non_afib_samples = non_afib_samples.drop(excess_non_afib.index)
        elif len(afib_samples) > len(non_afib_samples):
            # Excess AFIB samples — defer the entire record to phase 2
            excess_afib_samples.append(group_data)
            continue

        # Balanced group
        balanced_group = pd.concat([non_afib_samples, afib_samples])
        group_size = len(balanced_group)

        # Predicted proportions after adding the group
        total_current_samples = train_count + val_count + test_count

        train_dist, val_dist, test_dist = calculate_distances(
            train_count, val_count, test_count, group_size, total_current_samples, proportions
        )

        # Choose the best set based on minimum distance
        if train_dist <= val_dist and train_dist <= test_dist:
            train_set = pd.concat([train_set, balanced_group])
            train_count += group_size
            unused_samples["train"].append(excess_non_afib)
        elif val_dist <= train_dist and val_dist <= test_dist:
            val_set = pd.concat([val_set, balanced_group])
            val_count += group_size
            unused_samples["val"].append(excess_non_afib)
        else:
            test_set = pd.concat([test_set, balanced_group])
            test_count += group_size
            unused_samples["test"].append(excess_non_afib)

    # Phase 2: Processing excess AFIB samples
    def balance_with_non_afib(destination, excess_group, non_afib_needed):
        available_samples = pd.concat([sample for sample in unused_samples[destination]])
        balancing_samples = available_samples.sample(non_afib_needed, random_state=random_state)
        # Ensure only indices present in the DataFrame are dropped
        unused_samples[destination] = [
            sample.drop(sample.index.intersection(balancing_samples.index))
            for sample in unused_samples[destination]
        ]
        return pd.concat([balancing_samples, excess_group])

    for excess_group in tqdm(
        excess_afib_samples, desc="Processing excess AFIB samples in phase 2"
    ):
        # Calculate missing samples to balance the group
        non_afib_needed = len(excess_group[excess_group[stratify_col] == 1]) * 2 - len(
            excess_group
        )
        excess_afib_len_after_balancing = len(excess_group) + non_afib_needed

        total_current_samples = train_count + val_count + test_count

        train_dist, val_dist, test_dist = calculate_distances(
            train_count,
            val_count,
            test_count,
            excess_afib_len_after_balancing,
            total_current_samples,
            proportions,
        )

        # Choose the best set based on minimum distance
        if train_dist <= val_dist and train_dist <= test_dist:
            balanced_group = balance_with_non_afib("train", excess_group, non_afib_needed)
            train_set = pd.concat([train_set, balanced_group])
            train_count += len(balanced_group)
        elif val_dist <= train_dist and val_dist <= test_dist:
            balanced_group = balance_with_non_afib("val", excess_group, non_afib_needed)
            val_set = pd.concat([val_set, balanced_group])
            val_count += len(balanced_group)
        else:
            balanced_group = balance_with_non_afib("test", excess_group, non_afib_needed)
            test_set = pd.concat([test_set, balanced_group])
            test_count += len(balanced_group)

    return train_set, val_set, test_set


def process_dataset(
    test_size: float, val_size: float, verbosity: str, temp_file_path: str = "temp_results.npz"
) -> None:
    """
    Split the dataset into balanced training, validation, and test sets.

    :param test_size: Proportion of the dataset to include in the test split.
    :param val_size: Proportion of the dataset to include in the validation split.
    :param verbosity: Verbosity level for logging.
    :param temp_file_path: Path to the temporary file for saving/loading results.
    """
    logger.remove()
    logger.add(sys.stderr, level=verbosity.upper())
    logger.info("Starting dataset processing...")

    dataset_dir = INTERIM_DATA_DIR
    afib_files = {}
    non_afib_files = {}

    use_tqdm = verbosity.upper() != "DEBUG"

    afib_count = 0
    total_count = 0

    for root, _, files in os.walk(dataset_dir):
        if use_tqdm:
            files = tqdm(files, desc=f"Processing files in {root}")
        for file in files:
            if file.endswith(".npz"):
                file_path = os.path.join(root, file)
                dataset_name = os.path.basename(root)
                record_name = file.split("#")[1]
                unique_record_name = f"{dataset_name}#{record_name}"
                if unique_record_name not in afib_files:
                    afib_files[unique_record_name] = []
                if unique_record_name not in non_afib_files:
                    non_afib_files[unique_record_name] = []
                data = np.load(file_path)
                if np.any(data["y"] == 1):
                    afib_files[unique_record_name].append(file_path)
                    afib_count += 1
                else:
                    non_afib_files[unique_record_name].append(file_path)
                total_count += 1

    logger.info("Files categorized into AFIB and non-AFIB.")
    afib_percentage = (afib_count / total_count) * 100 if total_count > 0 else 0
    logger.info(f"Total files: {total_count}, AFIB percentage: {afib_percentage:.2f}%")

    records = []
    for record_name, files in afib_files.items():
        for file in files:
            records.append((file, 1, record_name))
    for record_name, files in non_afib_files.items():
        for file in files:
            records.append((file, 0, record_name))

    df = pd.DataFrame(records, columns=["file_path", "label", "group_id"])

    proportions = (1 - test_size - val_size, val_size, test_size)
    train_set, val_set, test_set = iterative_group_balanced_split(
        df, stratify_col="label", group_col="group_id", proportions=proportions
    )

    train_files = train_set["file_path"].tolist()
    val_files = val_set["file_path"].tolist()
    test_files = test_set["file_path"].tolist()

    logger.info("Files assigned to training, validation, and test sets.")

    def copy_files(file_list, target_dir):
        os.makedirs(target_dir, exist_ok=True)
        for file in tqdm(file_list, desc=f"Copying files to {target_dir}"):
            shutil.copy(file, target_dir)
        logger.info(f"Copied {len(file_list)} files to {target_dir}.")

    copy_files(train_files, TRAIN_DATA_DIR)
    copy_files(val_files, VAL_DATA_DIR)
    copy_files(test_files, TEST_DATA_DIR)

    def log_ratios(file_list, label):
        afib_count = 0
        for file in tqdm(file_list, desc=f"Calculating ratios for {label} set"):
            if np.any(np.load(file)["y"] == 1):
                afib_count += 1
        non_afib_count = len(file_list) - afib_count
        total_count = len(file_list)
        logger.info(
            f"{label} set: Total files: {total_count}, AFIB: {afib_count} ({(afib_count / total_count) * 100:.2f}%), "
            f"Non-AFIB: {non_afib_count} ({(non_afib_count / total_count) * 100:.2f}%)"
        )

    log_ratios(train_files, "Training")
    log_ratios(val_files, "Validation")
    log_ratios(test_files, "Test")

    logger.info(
        f"Data split into training, validation, and test sets and copied to respective directories."
    )


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
