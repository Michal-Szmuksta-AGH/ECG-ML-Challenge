from typing import Iterable, Union

import matplotlib.pyplot as plt
import numpy as np
import wfdb


def plot_ecg(
    record: Union[np.ndarray, wfdb.Record],
    fs: int,
    channels: Union[Iterable[int], int] = -1,
    t_start: int = 0,
    t_end: Union[int, None] = None,
    annotation: Union[wfdb.Annotation, None] = None,
) -> plt.Figure:
    """
    Plot ECG signal.

    :param record: Numpy array ECG signal or wfdb.Record object.
    :param fs: Sampling frequency.
    :param channels: List of channels to plot. If -1, plot all channels.
    :param t_start: Start time of the plot.
    :param t_end: End time of the plot. If None, plot the whole signal.
    :param annotation: Annotation object.
    """
    if not isinstance(record, (np.ndarray, wfdb.Record)):
        raise ValueError("record must be a numpy array or wfdb.Record object")
    if not isinstance(fs, int):
        raise ValueError("fs must be an integer")
    if not isinstance(channels, (Iterable, int)):
        raise ValueError("channels must be an iterable or integer")
    if not isinstance(t_start, int):
        raise ValueError("t_start must be an integer")
    if t_end is not None and not isinstance(t_end, int):
        raise ValueError("t_end must be an integer")
    if annotation is not None and not isinstance(annotation, wfdb.Annotation):
        raise ValueError("annotation must be a wfdb.Annotation object")

    if isinstance(record, wfdb.Record):
        signal = record.p_signal
    else:
        signal = record

    if t_end is None:
        num_samples = signal.shape[0] - t_start * fs
        t_end = signal.shape[0] / fs
    else:
        num_samples = int((t_end - t_start) * fs)

    time = np.linspace(t_start, t_end, num_samples)
    if channels == -1:
        try:
            channels = np.arange(signal.shape[1])
        except IndexError:
            channels = [0]
        num_channels = len(channels)
    else:
        num_channels = len(channels)

    sig_name = record.sig_name if isinstance(record, wfdb.Record) else None

    fig, ax = plt.subplots(
        num_channels, 1, figsize=(10, 4 * num_channels), sharex=True, sharey=True
    )
    if num_channels == 1:
        ax = [ax]
        signal = signal.reshape(-1, 1)
    fig.suptitle("ECG signal")
    fig.patch.set_facecolor("#ffe7e7")
    for i, channel in enumerate(channels):
        ax[i].plot(time, signal[t_start * fs : t_start * fs + num_samples, channel], color="blue")
        ax[i].set_ylabel("Amplitude")
        ax[i].set_xlim(t_start, t_end)
        ax[i].set_xticks(np.linspace(t_start, t_end, num=10))
        ax[i].set_title(f"CH{channel}{': ' + sig_name[channel] if sig_name is not None else ''}")
        ax[i].set_facecolor("#ffe7e7")
        ax[i].grid(color="red", linewidth=0.5)

        if annotation is not None:
            mask = (annotation.sample >= t_start * fs) & (annotation.sample < t_end * fs)
            ann_samples = annotation.sample[mask]
            ann_aux_notes = np.array(annotation.aux_note)[mask]

            y_pos = signal[ann_samples, channel] + np.max(signal[:, channel]) * 0.3

            for sample, aux_note, y in zip(ann_samples, ann_aux_notes, y_pos):
                if aux_note != "(N":
                    ax[i].annotate(
                        aux_note,
                        (sample / fs, y),
                        xytext=(0, 10),
                        textcoords="offset points",
                        ha="center",
                        color="red",
                    )

            j = 0
            while j < len(ann_samples):
                if ann_aux_notes[j] == "(AFIB":
                    k = j + 1
                    while k < len(ann_samples):
                        if ann_aux_notes[k] != "(AFIB":
                            ax[i].axvspan(
                                ann_samples[j] / fs,
                                ann_samples[k] / fs,
                                color="red",
                                alpha=0.2,
                            )
                            j = k
                            break
                        k += 1
                j += 1

    ax[-1].set_xlabel("Time [s]")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    return fig
