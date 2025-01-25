import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from loguru import logger
from src.model.models import get_model
from src.dataset.dataloaders import ECGDataset
from src.config import FIGURES_DIR
import numpy as np
from tqdm import tqdm
from src.config import TRAIN_DATA_DIR, MODELS_DIR, RAW_DATA_DIR
from src.model.models import add_hooks
import wfdb
from wfdb import processing
from src.dataset.processingv2 import (
    resample_data_and_annotations,
    get_channel_for_qrs_extraction,
    find_afib_segments,
)


def select_random_samples_with_both_classes(test_data_loader: DataLoader, num_samples: int):
    """
    Select a specified number of random samples from the test dataset that contain both classes (0 and 1).

    :param test_data_loader: DataLoader for the test dataset.
    :param num_samples: Number of samples to select.
    :return: List of selected sample data, labels, and filenames.
    """
    samples = list(test_data_loader)
    random.shuffle(samples)
    selected_samples = []
    for idx, (data, label) in enumerate(samples):
        if 0 in label and 1 in label:
            selected_samples.append((data, label, test_data_loader.dataset.files[idx]))
            if len(selected_samples) == num_samples:
                break
    if len(selected_samples) < num_samples:
        raise ValueError(
            f"Only found {len(selected_samples)} samples containing both classes in the test dataset."
        )
    return selected_samples


def remove_prefix_from_state_dict(state_dict, prefix):
    """
    Remove prefix from state_dict keys if they exist.

    :param state_dict: The state dictionary of the model.
    :param prefix: The prefix to remove.
    :return: The state dictionary with prefixes removed.
    """
    return {k[len(prefix) :] if k.startswith(prefix) else k: v for k, v in state_dict.items()}


def plot_comparative_graph(
    sample_data, sample_label, predicted_label, sample_filename, probabilities
):
    """
    Plot comparative graphs of the sample, showing the signal, true labels, predicted labels, and probabilities.

    :param sample_data: The ECG signal data.
    :param sample_label: The true labels.
    :param predicted_label: The predicted labels by the model.
    :param sample_filename: The filename of the sample.
    :param probabilities: The predicted probabilities by the model.
    """
    sample_data = sample_data.cpu().numpy().flatten()
    sample_label = sample_label.cpu().numpy().flatten()
    predicted_label = predicted_label.cpu().numpy().flatten()
    probabilities = probabilities.cpu().numpy().flatten()

    plt.figure(figsize=(15, 10))

    # Plot ECG Signal
    plt.subplot(3, 1, 1)
    plt.plot(sample_data, label="ECG Signal")
    plt.title("ECG Signal")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.legend()

    # Plot True and Predicted Labels with TP, TN, FP, FN
    plt.subplot(3, 1, 2)
    tp_indices = (predicted_label == 1) & (sample_label == 1)
    tn_indices = (predicted_label == 0) & (sample_label == 0)
    fp_indices = (predicted_label == 1) & (sample_label == 0)
    fn_indices = (predicted_label == 0) & (sample_label == 1)

    plt.eventplot(np.where(tp_indices), lineoffsets=0, colors="green", label="TP")
    plt.eventplot(np.where(tn_indices), lineoffsets=0, colors="lightgreen", label="TN")
    plt.eventplot(np.where(fp_indices), lineoffsets=0, colors="red", label="FP")
    plt.eventplot(np.where(fn_indices), lineoffsets=0, colors="lightcoral", label="FN")

    plt.title("True and Predicted Labels with TP, TN, FP, FN")
    plt.xlabel("Sample Index")
    plt.ylabel("Event")
    plt.legend()

    # Plot Probabilities and Threshold
    plt.subplot(3, 1, 3)
    plt.plot(probabilities, label="Predicted Probabilities", color="blue")
    plt.axhline(y=0.5, color="r", linestyle="--", label="Threshold (0.5)")
    plt.title("Predicted Probabilities and Threshold")
    plt.xlabel("Sample Index")
    plt.ylabel("Probability")
    plt.legend()

    plt.tight_layout()

    os.makedirs(FIGURES_DIR, exist_ok=True)
    file_name = os.path.splitext(os.path.basename(sample_filename))[0]
    file_path = os.path.join(FIGURES_DIR, f"{file_name}_comparative_graph.png")
    plt.savefig(file_path)
    logger.info(f"Comparative graph saved as '{file_path}'")


def evaluate_model(
    model_type: str,
    state_dict_name: str,
    test_data_dir: str,
    models_dataset_dir: str,
    num_samples: int = 1,
) -> None:
    """
    Evaluate the trained model on a specified number of random samples from the test dataset.

    :param model_type: Type of the model.
    :param state_dict_name: Name of the state_dict file.
    :param test_data_dir: Directory of the test dataset.
    :param models_dataset_dir: Directory of the models dataset.
    :param num_samples: Number of samples to evaluate.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(model_type).to(device)
    state_dict_path = os.path.join(models_dataset_dir, state_dict_name)
    state_dict = torch.load(state_dict_path, weights_only=True)
    state_dict = remove_prefix_from_state_dict(state_dict, "_orig_mod.")

    model.load_state_dict(state_dict)
    model.eval()

    test_dataset = ECGDataset(test_data_dir)
    test_data_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    selected_samples = select_random_samples_with_both_classes(test_data_loader, num_samples)

    for sample_data, sample_label, sample_filename in tqdm(
        selected_samples, desc="Evaluating samples"
    ):
        sample_data = sample_data.to(device)
        sample_label = sample_label.to(device)

        with torch.no_grad():
            output = model(sample_data)
            probabilities = torch.sigmoid(output)
            predicted_label = (probabilities > 0.5).int()

        tp = ((predicted_label == 1) & (sample_label == 1)).sum().item()
        tn = ((predicted_label == 0) & (sample_label == 0)).sum().item()
        fp = ((predicted_label == 1) & (sample_label == 0)).sum().item()
        fn = ((predicted_label == 0) & (sample_label == 1)).sum().item()

        accuracy = (tp + tn) / (tp + tn + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = (
            2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        )

        logger.info(f"Sample filename: {sample_filename}")
        logger.info(f"Shape of sample_data: {sample_data.shape}")
        logger.info(f"Shape of sample_label: {sample_label.shape}")
        logger.info(f"Shape of predicted_label: {predicted_label.shape}")
        logger.info(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
        logger.info(
            f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1_score:.4f}"
        )

        plot_comparative_graph(
            sample_data, sample_label, predicted_label, sample_filename, probabilities
        )


def evaluate_model_v2(
    model_type: str,
    state_dict_name: str,
    model_chunk_size: int,
    model_fs: int,
    models_dataset_dir: str,
    database: str,
    record: str,
    xlim_min: int = None,  # New argument
    xlim_max: int = None,  # New argument
):
    model_path = os.path.join(models_dataset_dir, state_dict_name)
    _model = get_model(model_type)
    state_dict = torch.load(model_path, weights_only=True, map_location="cpu")
    prefix = "_orig_mod."
    state_dict = {
        k[len(prefix) :] if k.startswith(prefix) else k: v for k, v in state_dict.items()
    }
    _model.load_state_dict(state_dict)
    _model.eval()

    record_path = RAW_DATA_DIR / database / record
    record_data = wfdb.rdrecord(str(record_path))
    try:
        annotation = wfdb.rdann(str(record_path), "atr")
    except:
        annotation = None
    qrs_annotation_path = record_path.with_suffix(".qrs")
    if qrs_annotation_path.exists():
        qrs_annotation = wfdb.rdann(str(record_path), "qrs")
    else:
        qrs_annotation = None

    resampled_record, resampled_annotation, resampled_qrs_annotation = (
        resample_data_and_annotations(record_data, annotation, qrs_annotation, model_fs)
    )
    channel = get_channel_for_qrs_extraction(database)
    afib_segments = find_afib_segments(resampled_annotation)

    fs = resampled_annotation.fs
    signal = resampled_record
    signal = signal.astype(np.float32)
    signal = signal[..., channel]

    # QRS and RR interval detection
    xqrs = wfdb.processing.XQRS(sig=signal, fs=fs)
    xqrs.detect()
    qrs_inds = xqrs.qrs_inds
    qrs_inds = qrs_inds.astype(np.int32)
    qrs_inds = processing.correct_peaks(
        signal, qrs_inds, search_radius=int(0.1 * fs), smooth_window_size=150
    )
    rr = wfdb.processing.calc_rr(
        qrs_inds, fs=fs, min_rr=None, max_rr=None, qrs_units="samples", rr_units="samples"
    )

    input_rr_samples = model_chunk_size
    batch_size = 60
    qrs_af_probabs = np.zeros(shape=(len(qrs_inds),), dtype=np.float32)
    qrs_af_overlap = np.zeros(shape=(len(qrs_inds),), dtype=np.float32)

    pred_step = 10
    batch = np.zeros(shape=(batch_size, input_rr_samples, 1), dtype=np.float32)
    batch_idx = 0
    rr_indices_history = []
    for rr_idx in tqdm(
        range(0, rr.shape[0] - input_rr_samples, pred_step), desc="Processing RR intervals"
    ):
        snippet = rr[rr_idx : rr_idx + input_rr_samples]
        rr_indices_history.append([rr_idx, rr_idx + input_rr_samples])
        snippet = snippet[..., np.newaxis]
        batch[batch_idx] = snippet
        batch_idx += 1

        if batch_idx == batch_size:
            with torch.no_grad():
                results = _model(torch.from_numpy(batch).float()).numpy()
            for j in range(batch_idx):
                rr_from, rr_to = rr_indices_history[j]
                qrs_af_probabs[rr_from:rr_to] += results[j, :, 0]
                qrs_af_overlap[rr_from:rr_to] += 1.0

            batch_idx = 0
            rr_indices_history = []

    if batch_idx > 0:
        with torch.no_grad():
            results = _model(torch.from_numpy(batch).float()).numpy()
        for j in range(batch_idx):
            rr_from, rr_to = rr_indices_history[j]
            qrs_af_probabs[rr_from:rr_to] += results[j, :, 0]
            qrs_af_overlap[rr_from:rr_to] += 1.0

    qrs_af_overlap[qrs_af_overlap == 0.0] = 1.0
    qrs_af_probabs /= qrs_af_overlap
    qrs_af_preds = np.round(qrs_af_probabs)

    pred = np.zeros(
        [
            len(signal),
        ],
        dtype=np.float32,
    )
    probs = np.zeros(
        [
            len(signal),
        ],
        dtype=np.float32,
    )

    for qrs_idx in tqdm(range(len(rr)), desc="Assigning predictions to signal"):
        pred[qrs_inds[qrs_idx] : qrs_inds[qrs_idx + 1]] = qrs_af_preds[qrs_idx]
        probs[qrs_inds[qrs_idx] : qrs_inds[qrs_idx + 1]] = qrs_af_probabs[qrs_idx]

    plt.figure(figsize=(15, 10))

    # Plot ECG Signal with AFib segments highlighted
    plt.subplot(2, 1, 1)
    plt.plot(signal, label="ECG Signal")
    # Fill background from start to first AFib and from last AFib to end
    if afib_segments:
        plt.axvspan(0, afib_segments[0][0], color="lightgreen", alpha=0.5)
        plt.axvspan(afib_segments[-1][1], len(signal), color="lightgreen", alpha=0.5)

    for start, end in tqdm(afib_segments, desc="Plotting AFib segments"):
        plt.axvspan(
            start,
            end,
            color="lightcoral",
            alpha=0.5,
            label="AFib" if start == afib_segments[0][0] else "",
        )
        if afib_segments.index((start, end)) < len(afib_segments) - 1:
            next_start = afib_segments[afib_segments.index((start, end)) + 1][0]
            plt.axvspan(
                end,
                next_start,
                color="lightgreen",
                alpha=0.5,
                label="Non-AFib" if start == afib_segments[0][0] else "",
            )
    plt.title("ECG Signal with AFib Segments")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.legend()
    if xlim_min is not None and xlim_max is not None:
        plt.xlim(xlim_min, xlim_max)

    # Plot Probabilities with predicted AFib segments highlighted
    plt.subplot(2, 1, 2)
    plt.plot(probs, label="Probabilities", color="blue")
    afib_indices = np.where(probs > 0.5)[0]
    non_afib_indices = np.where(probs <= 0.5)[0]

    afib_segments = []
    non_afib_segments = []

    # Group consecutive indices into segments
    for indices, segments in tqdm(
        [(afib_indices, afib_segments), (non_afib_indices, non_afib_segments)],
        desc="Grouping consecutive indices into segments",
    ):
        if len(indices) > 0:
            start = indices[0]
            for i in range(1, len(indices)):
                if indices[i] != indices[i - 1] + 1:
                    segments.append((start, indices[i - 1] + 1))
                    start = indices[i]
            segments.append((start, indices[-1] + 1))

    for start, end in tqdm(afib_segments, desc="Plotting Predicted AFib segments"):
        plt.axvspan(
            start,
            end,
            color="lightcoral",
            alpha=0.5,
            label="Predicted AFib" if start == afib_segments[0][0] else "",
        )
    for start, end in non_afib_segments:
        plt.axvspan(
            start,
            end,
            color="lightgreen",
            alpha=0.5,
            label="Predicted Non-AFib" if start == non_afib_segments[0][0] else "",
        )
    plt.title("Probabilities with Predicted AFib Segments")
    plt.xlabel("Sample Index")
    plt.ylabel("Probability")
    plt.legend()
    if xlim_min is not None and xlim_max is not None:
        plt.xlim(xlim_min, xlim_max)

    plt.tight_layout()

    os.makedirs(FIGURES_DIR, exist_ok=True)
    file_name = f"{record}_evaluation_graph.png"
    file_path = os.path.join(FIGURES_DIR, file_name)
    plt.savefig(file_path)
    logger.info(f"Evaluation graph saved as '{file_path}'")


def evaluate_tensor_shapes(
    model_type: str, batch_size: int = 8, models_dataset_dir: str = MODELS_DIR
) -> None:
    """
    Evaluate tensor shapes during model processing for a random batch from the training dataset.

    :param model_type: Type of the model.
    :param batch_size: Size of the batch to evaluate.
    :param models_dataset_dir: Directory of the models dataset.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(model_type).to(device)
    model.eval()

    hooks = add_hooks(model)

    train_dataset = ECGDataset(TRAIN_DATA_DIR)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    sample_data, sample_label = next(iter(train_data_loader))
    sample_data = sample_data.to(device)

    try:
        with torch.no_grad():
            model(sample_data)
    except Exception as e:
        logger.error(f"Dimension error during model evaluation: {str(e)}")
        return  # Stop the program after the first encountered dimension error

    for hook in hooks:
        if hook.input is not None and hook.output is not None:
            if isinstance(hook.input, torch.Tensor):
                input_shape = hook.input.shape
            else:
                input_shape = hook.input[0].shape
            if isinstance(hook.output, torch.Tensor):
                output_shape = hook.output.shape
            else:
                output_shape = hook.output[0].shape
            print(f"Module: {hook.module}")
            print(f"Input shape: {input_shape}")
            print(f"Output shape: {output_shape}\n")
        else:
            print(f"Error in module: {hook.module}")
            return  # Stop the program after the first encountered dimension error
        print("-" * 51)  # 51 dashes to fill the line

    for hook in hooks:
        hook.close()
