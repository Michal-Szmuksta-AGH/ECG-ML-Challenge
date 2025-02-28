from src.config import (
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    TEST_DATA_DIR,
    TRAIN_DATA_DIR,
    VAL_DATA_DIR,
)

from src.dataset.processing import download_wfdb_dataset

import os
import sys
import numpy as np
import wfdb
from loguru import logger
from tqdm import tqdm
from wfdb import processing
from sklearn.model_selection import train_test_split
import shutil


def preprocess_dataset_v2(
    dataset_name: str, target_fs: int, chunk_size: int, step: int, verbosity: str
) -> None:
    """
    Process the entire dataset with QRS feature extraction.

    :param dataset_name: Name of the dataset.
    :param target_fs: Target sampling frequency.
    :param chunk_size: Number of samples per chunk.
    :param step: Step of window step while creating chunks.
    :param verbosity: Verbosity level for logging.
    """
    logger.remove()
    logger.add(sys.stderr, level=verbosity.upper())

    # Download the dataset
    dataset_dir = os.path.join(RAW_DATA_DIR, dataset_name)
    interim_data_dir = INTERIM_DATA_DIR
    download_wfdb_dataset(dataset_name, dataset_dir)

    # Prepare files for processing
    files = os.listdir(dataset_dir)
    files = [file.split(".")[0] for file in files if file.endswith(".hea")]
    files = reject_unacceptable_files(dataset_name, files)

    # Initialize other variables
    channel = get_channel_for_qrs_extraction(dataset_name)

    # Process each record
    logger.info(f"Preprocessing {len(files)} ECG records from {dataset_name}...")
    use_tqdm = verbosity.upper() != "DEBUG"
    file_iterator = (
        tqdm(files, desc=f"Preprocessing files from {dataset_name}") if use_tqdm else files
    )
    for file in file_iterator:
        # Load the record and annotations
        logger.debug(f"Loading {file} from {dataset_dir}...")
        record, annotation, qrs_annotation = load_record_and_annotations(
            dataset_name, dataset_dir, file
        )

        # Resample the record and annotations
        logger.debug(f"Resampling {file} to {target_fs} Hz...")
        resampled_record, resampled_annotation, resampled_qrs_annotation = (
            resample_data_and_annotations(record, annotation, qrs_annotation, target_fs)
        )

        # Get QRS locationsos.path.join(, correct them, and calculate RR intervals
        logger.debug(f"Extracting QRS features from {file}...")
        qrs_locs = get_qrs_locs(dataset_name, resampled_annotation, resampled_qrs_annotation)
        if resampled_record is not None:
            qrs_locs = correct_qrs_locs(resampled_record, channel, qrs_locs, target_fs)
        rr_intervals = calculate_rr(qrs_locs, target_fs)

        # Get the AFIB segments and mark RR intervals
        logger.debug(f"Marking RR intervals for {file}...")
        if resampled_annotation is not None:
            afib_segments = find_afib_segments(resampled_annotation)
        else:
            afib_segments = afib_segments = [
                (resampled_qrs_annotation.sample[0], resampled_qrs_annotation.sample[-1])
            ]
        labels = mark_rr_intervals(qrs_locs, afib_segments)

        # Split the record into chunks
        logger.debug(f"Splitting {file} into chunks of size {chunk_size}...")
        rr_interval_chunks, label_chunks = split_into_chunks_truncate(
            rr_intervals, labels, chunk_size, step
        )

        # Save the chunks
        logger.debug(f"Saving chunks of {file} to {interim_data_dir}...")
        os.makedirs(interim_data_dir, exist_ok=True)
        for i in range(len(rr_interval_chunks)):
            np.savez(
                os.path.join(interim_data_dir, f"{dataset_name}#{file}#{i}.npz"),
                x=rr_interval_chunks[i],
                y=label_chunks[i],
            )

    logger.info(f"All preprocessed ECG data from {dataset_name} saved to {interim_data_dir}")


def load_record_and_annotations(dataset_name, dataset_dir, file):
    if dataset_name == "mitdb":
        record = wfdb.rdrecord(os.path.join(dataset_dir, file))
        annotation = wfdb.rdann(os.path.join(dataset_dir, file), "atr")
        return record, annotation, None
    elif dataset_name == "afdb":
        try:
            record = wfdb.rdrecord(os.path.join(dataset_dir, file))
        except:
            record = None
        annotation = wfdb.rdann(os.path.join(dataset_dir, file), "atr")
        try:
            qrs_annotation = wfdb.rdann(os.path.join(dataset_dir, file), "qrsc")
        except:
            qrs_annotation = wfdb.rdann(os.path.join(dataset_dir, file), "qrs")
        return record, annotation, qrs_annotation
    elif dataset_name == "af-termination-challenge":
        record = wfdb.rdrecord(os.path.join(dataset_dir, file))
        qrs_annotation = wfdb.rdann(os.path.join(dataset_dir, file), "qrs")
        return record, None, qrs_annotation
    elif dataset_name == "2017_challenge":
        record = wfdb.rdrecord(os.path.join(dataset_dir, file))
        annotation = wfdb.rdann(os.path.join(dataset_dir, file), "atr")
        return record, annotation, None
    elif dataset_name == "long_term":
        record = wfdb.rdrecord(os.path.join(dataset_dir, file))
        annotation = wfdb.rdann(os.path.join(dataset_dir, file), "atr")
        qrs_annotation = wfdb.rdann(os.path.join(dataset_dir, file), "qrs")
        return record, annotation, qrs_annotation
    elif dataset_name == "shdb-af":
        record = wfdb.rdrecord(os.path.join(dataset_dir, file))
        annotation = wfdb.rdann(os.path.join(dataset_dir, file), "atr")
        qrs_annotation = wfdb.rdann(os.path.join(dataset_dir, file), "qrs")
        return record, annotation, qrs_annotation


def reject_unacceptable_files(dataset_name, files):
    if dataset_name == "mitdb":
        files = [file for file in files if file not in ["102", "104", "107", "217", "114"]]
    elif dataset_name == "afdb":
        pass
    elif dataset_name == "af-termination-challenge":
        pass
    elif dataset_name == "2017_challenge":
        # TODO można usunąć pliki z zaszumionymi danymi
        pass
    elif dataset_name == "long_term":
        pass
    elif dataset_name == "shdb-af":
        pass

    return files


def get_channel_for_qrs_extraction(dataset_name):
    if dataset_name == "mitdb":
        return 0
    elif dataset_name == "afdb":
        return 1
    elif dataset_name == "af-termination-challenge":
        return 0
    elif dataset_name == "2017_challenge":
        return 1
    elif dataset_name == "long_term":
        return 0
    elif dataset_name == "shdb-af":
        return 0


def get_qrs_locs(dataset_name, resampled_annotation, resampled_qrs_annotation):
    if dataset_name == "mitdb":
        qrs_locs = resampled_annotation.sample
    elif dataset_name == "afdb":
        qrs_locs = resampled_qrs_annotation.sample
    elif dataset_name == "af-termination-challenge":
        qrs_locs = resampled_qrs_annotation.sample
    elif dataset_name == "2017_challenge":
        qrs_locs = resampled_annotation.sample
    elif dataset_name == "long_term":
        qrs_locs = resampled_qrs_annotation.sample
    elif dataset_name == "shdb-af":
        qrs_locs = resampled_qrs_annotation.sample
    return qrs_locs


def resample_data_and_annotations(record_data, annotation, qrs_annotation, target_fs):
    if record_data is None:
        signal = np.zeros((10, 1))
        try:
            fs = annotation.fs
        except:
            try:
                fs = qrs_annotation.fs
            except:
                raise ValueError("No fs attribute found in the annotations")
    else:
        signal = record_data.p_signal
        fs = record_data.fs

    if annotation is None:
        note = wfdb.Annotation("xd", "atr", np.zeros((10, 1)))
    else:
        note = annotation

    resampled_record, resampled_annotation = processing.resample_multichan(
        signal, note, fs, target_fs
    )

    if qrs_annotation is not None:
        _, resampled_qrs_annotation = processing.resample_multichan(
            signal, qrs_annotation, fs, target_fs
        )
    else:
        resampled_qrs_annotation = None

    resampled_record = resampled_record if record_data is not None else None
    resampled_annotation = resampled_annotation if annotation is not None else None
    return resampled_record, resampled_annotation, resampled_qrs_annotation


def correct_qrs_locs(signal, channel, qrs_locs, fs):
    qrs_indices = processing.correct_peaks(
        signal[:, channel], qrs_locs, search_radius=int(0.1 * fs), smooth_window_size=150
    )
    return qrs_indices


def calculate_rr(qrs_locs, fs):
    rr_intervals = wfdb.processing.calc_rr(
        qrs_locs, fs=fs, min_rr=None, max_rr=None, qrs_units="samples", rr_units="seconds"
    )
    return rr_intervals


def find_afib_segments(annotation):
    afib_segments = []
    in_afib = False
    start = None

    for i, note in enumerate(annotation.aux_note):
        if note == "" or note == "None":
            continue

        if "AFIB" in note and not in_afib:
            start = annotation.sample[i]
            in_afib = True

        elif "AFIB" not in note and in_afib:
            end = annotation.sample[i]
            afib_segments.append((start, end))
            in_afib = False

    if in_afib:
        afib_segments.append((start, annotation.sample[-1]))

    return afib_segments


def mark_rr_intervals(qrs_locs, afib_segments):
    afib_labels = []

    for i in range(len(qrs_locs) - 1):
        start, end = qrs_locs[i], qrs_locs[i + 1]
        total_samples = end - start

        for afib_start, afib_end in afib_segments:
            overlap_start = max(start, afib_start)
            overlap_end = min(end, afib_end)

            if overlap_start < overlap_end:  # Overlap exists
                afib_samples = overlap_end - overlap_start
                if afib_samples > total_samples / 2:
                    afib_labels.append(1)
                    break
        else:
            afib_labels.append(0)

    return np.array(afib_labels).reshape(-1, 1)


def split_into_chunks_truncate(rr_intervals, labels, chunk_size, step):
    n = len(rr_intervals)
    rr_chunks = []
    label_chunks = []

    # Tworzenie okien o przesuwie "step"
    for i in range(0, n - chunk_size + 1, step):
        rr_chunks.append(rr_intervals[i : i + chunk_size])
        label_chunks.append(labels[i : i + chunk_size])

    # Konwersja list na tablice numpy
    rr_chunks = np.array(rr_chunks)
    label_chunks = np.array(label_chunks)

    return rr_chunks.squeeze(), label_chunks.squeeze()


def process_dataset_v2(test_size: float, val_size: float, verbosity: str) -> None:
    """
    Split the dataset into balanced training, validation, and test sets.

    :param test_size: Proportion of the dataset to include in the test split.
    :param val_size: Proportion of the dataset to include in the validation split.
    :param verbosity: Verbosity level for logging.
    """
    logger.remove()
    logger.add(sys.stderr, level=verbosity.upper())

    # Prepare the directories
    interim_data_dir = INTERIM_DATA_DIR
    train_data_dir = TRAIN_DATA_DIR
    test_data_dir = TEST_DATA_DIR
    val_data_dir = VAL_DATA_DIR
    os.makedirs(train_data_dir, exist_ok=True)
    os.makedirs(test_data_dir, exist_ok=True)
    os.makedirs(val_data_dir, exist_ok=True)

    # Get all .npz files in the interim_data_dir
    npz_files = [f for f in os.listdir(interim_data_dir) if f.endswith(".npz")]

    # Get the statistics of AFIB labels in each file
    afib_samples, non_afib_samples = get_samples_statistics(interim_data_dir, npz_files)
    record_statistics = get_record_statistics(npz_files, afib_samples, non_afib_samples)

    # Split the dataset into training, validation, and test sets
    train_size = 1 - test_size - val_size
    train_records, test_records, val_records, statistics = greedy_record_split(
        record_statistics, train_size, test_size, val_size
    )

    # Log the statistics of the datasets
    log_datasets_stats(statistics)

    # Put the chunk files into the respective directories
    train_files, test_files, val_files = assign_files_to_datasets(
        npz_files, train_records, test_records, val_records
    )

    # Copy the files to the respective directories
    logger.info("Copying files to the respective directories...")
    use_tqdm = verbosity.upper() != "DEBUG"

    if train_files:
        train_iterator = (
            tqdm(train_files, desc="Copying training files") if use_tqdm else train_files
        )
        for file in train_iterator:
            shutil.copy(os.path.join(interim_data_dir, file), os.path.join(train_data_dir, file))

    if val_files:
        val_iterator = tqdm(val_files, desc="Copying validation files") if use_tqdm else val_files
        for file in val_iterator:
            shutil.copy(os.path.join(interim_data_dir, file), os.path.join(val_data_dir, file))

    if test_files:
        test_iterator = tqdm(test_files, desc="Copying test files") if use_tqdm else test_files
        for file in test_iterator:
            shutil.copy(os.path.join(interim_data_dir, file), os.path.join(test_data_dir, file))


def get_samples_statistics(dir, files):
    afib_samples = []
    non_afib_samples = []

    for npz_file in tqdm(files, desc="Calculating samples statistics"):
        data = np.load(os.path.join(dir, npz_file))
        labels = data["y"]
        sum = np.sum(labels)
        length = len(labels)
        afib_samples.append(sum)
        non_afib_samples.append(length - sum)

    return afib_samples, non_afib_samples


def get_record_statistics(npz_files, afib_samples, non_afib_samples):
    record_stats = {}

    for file, afib, non_afib in tqdm(
        zip(npz_files, afib_samples, non_afib_samples), desc="Calculating record statistics"
    ):
        prefix = "#".join(file.split("#")[:2])
        if prefix not in record_stats:
            record_stats[prefix] = {"length": 0, "afib_samples": 0, "non_afib_samples": 0}

        record_stats[prefix]["length"] += 1
        record_stats[prefix]["afib_samples"] += afib
        record_stats[prefix]["non_afib_samples"] += non_afib

    return record_stats


def greedy_record_split(record_statistics, train_size, test_size, val_size):
    def proportion_loss(curr_train_size, curr_test_size, curr_val_size):
        curr_total = curr_train_size + curr_test_size + curr_val_size
        curr_train_prop = curr_train_size / curr_total
        curr_test_prop = curr_test_size / curr_total
        curr_val_prop = curr_val_size / curr_total

        train_prop_loss = (curr_train_prop - train_size) ** 2
        test_prop_loss = (curr_test_prop - test_size) ** 2
        val_prop_loss = (curr_val_prop - val_size) ** 2

        return train_prop_loss + test_prop_loss + val_prop_loss

    def distribution_loss(
        curr_train_afib_samples,
        curr_train_non_afib_samples,
        curr_test_afib_samples,
        curr_test_non_afib_samples,
        curr_val_afib_samples,
        curr_val_non_afib_samples,
    ):
        curr_train_total = curr_train_afib_samples + curr_train_non_afib_samples
        curr_test_total = curr_test_afib_samples + curr_test_non_afib_samples
        curr_val_total = curr_val_afib_samples + curr_val_non_afib_samples

        curr_train_afib_prop = (
            curr_train_afib_samples / curr_train_total if curr_train_total != 0 else 0
        )
        curr_test_afib_prop = (
            curr_test_afib_samples / curr_test_total if curr_test_total != 0 else 0
        )
        curr_val_afib_prop = curr_val_afib_samples / curr_val_total if curr_val_total != 0 else 0

        afib_distribution_loss = (
            (curr_train_afib_prop - curr_test_afib_prop) ** 2
            + (curr_train_afib_prop - curr_val_afib_prop) ** 2
            + (curr_test_afib_prop - curr_val_afib_prop) ** 2
        )

        return afib_distribution_loss

    def loss(
        curr_train_size,
        curr_test_size,
        curr_val_size,
        curr_train_afib_samples,
        curr_train_non_afib_samples,
        curr_test_afib_samples,
        curr_test_non_afib_samples,
        curr_val_afib_samples,
        curr_val_non_afib_samples,
    ):
        return 0.5 * proportion_loss(
            curr_train_size, curr_test_size, curr_val_size
        ) + 0.5 * distribution_loss(
            curr_train_afib_samples,
            curr_train_non_afib_samples,
            curr_test_afib_samples,
            curr_test_non_afib_samples,
            curr_val_afib_samples,
            curr_val_non_afib_samples,
        )

    train_prefixes = []
    test_prefixes = []
    val_prefixes = []
    curr_train_size = curr_test_size = curr_val_size = 0
    curr_train_afib_samples = curr_train_non_afib_samples = 0
    curr_test_afib_samples = curr_test_non_afib_samples = 0
    curr_val_afib_samples = curr_val_non_afib_samples = 0

    prefix_iterator = tqdm(record_statistics.items(), desc="Splitting records")
    for prefix, stats in prefix_iterator:
        prefix_length = stats["length"]
        prefix_afib_samples = stats["afib_samples"]
        prefix_non_afib_samples = stats["non_afib_samples"]

        if train_size == 0:
            train_loss = float("inf")
        else:
            train_loss = loss(
                curr_train_size + prefix_length,
                curr_test_size,
                curr_val_size,
                curr_train_afib_samples + prefix_afib_samples,
                curr_train_non_afib_samples + prefix_non_afib_samples,
                curr_test_afib_samples,
                curr_test_non_afib_samples,
                curr_val_afib_samples,
                curr_val_non_afib_samples,
            )

        if test_size == 0:
            test_loss = float("inf")
        else:
            test_loss = loss(
                curr_train_size,
                curr_test_size + prefix_length,
                curr_val_size,
                curr_train_afib_samples,
                curr_train_non_afib_samples,
                curr_test_afib_samples + prefix_afib_samples,
                curr_test_non_afib_samples + prefix_non_afib_samples,
                curr_val_afib_samples,
                curr_val_non_afib_samples,
            )

        if val_size == 0:
            val_loss = float("inf")
        else:
            val_loss = loss(
                curr_train_size,
                curr_test_size,
                curr_val_size + prefix_length,
                curr_train_afib_samples,
                curr_train_non_afib_samples,
                curr_test_afib_samples,
                curr_test_non_afib_samples,
                curr_val_afib_samples + prefix_afib_samples,
                curr_val_non_afib_samples + prefix_non_afib_samples,
            )

        if train_loss <= test_loss and train_loss <= val_loss:
            train_prefixes.append(prefix)
            curr_train_size += prefix_length
            curr_train_afib_samples += prefix_afib_samples
            curr_train_non_afib_samples += prefix_non_afib_samples
        elif test_loss <= train_loss and test_loss <= val_loss:
            test_prefixes.append(prefix)
            curr_test_size += prefix_length
            curr_test_afib_samples += prefix_afib_samples
            curr_test_non_afib_samples += prefix_non_afib_samples
        elif val_loss <= train_loss and val_loss <= test_loss:
            val_prefixes.append(prefix)
            curr_val_size += prefix_length
            curr_val_afib_samples += prefix_afib_samples
            curr_val_non_afib_samples += prefix_non_afib_samples

    stats = {
        "train_size": curr_train_size,
        "test_size": curr_test_size,
        "val_size": curr_val_size,
        "train_afib_samples": curr_train_afib_samples,
        "train_non_afib_samples": curr_train_non_afib_samples,
        "test_afib_samples": curr_test_afib_samples,
        "test_non_afib_samples": curr_test_non_afib_samples,
        "val_afib_samples": curr_val_afib_samples,
        "val_non_afib_samples": curr_val_non_afib_samples,
    }

    return train_prefixes, test_prefixes, val_prefixes, stats


def log_datasets_stats(stats):
    train_all_samples = stats["train_afib_samples"] + stats["train_non_afib_samples"]
    train_afib_prop = (
        stats["train_afib_samples"] / train_all_samples if train_all_samples != 0 else 0
    )
    test_all_samples = stats["test_afib_samples"] + stats["test_non_afib_samples"]
    test_afib_prop = stats["test_afib_samples"] / test_all_samples if test_all_samples != 0 else 0
    val_all_samples = stats["val_afib_samples"] + stats["val_non_afib_samples"]
    val_afib_prop = stats["val_afib_samples"] / val_all_samples if val_all_samples != 0 else 0

    total_size = stats["train_size"] + stats["test_size"] + stats["val_size"]
    train_prop = stats["train_size"] / total_size if total_size != 0 else 0
    test_prop = stats["test_size"] / total_size if total_size != 0 else 0
    val_prop = stats["val_size"] / total_size if total_size != 0 else 0

    logger.info(f"Training set AFIB proportion: {train_afib_prop:.2f}")
    logger.info(f"Test set AFIB proportion: {test_afib_prop:.2f}")
    logger.info(f"Validation set AFIB proportion: {val_afib_prop:.2f}")

    logger.info(f"Training set proportion: {train_prop:.2f}")
    logger.info(f"Test set proportion: {test_prop:.2f}")
    logger.info(f"Validation set proportion: {val_prop:.2f}")


def assign_files_to_datasets(npz_files, train_records, test_records, val_records):
    train_files = []
    test_files = []
    val_files = []

    npz_iterator = tqdm(npz_files, desc="Assigning files to datasets")
    for file in npz_iterator:
        prefix = "#".join(file.split("#")[:2])
        if prefix in train_records:
            train_files.append(file)
        elif prefix in test_records:
            test_files.append(file)
        elif prefix in val_records:
            val_files.append(file)
        else:
            raise ValueError(f"File {file} does not belong to any dataset")

    return train_files, test_files, val_files
