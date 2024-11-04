import wfdb
import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Iterable

def plot_ecg(record: Union[np.ndarray, wfdb.Record], fs: int, channels: Union[Iterable[int], int] = -1, duration: int = 10) -> plt.Figure:
    """
    Plot ECG signal.

    :param record: Numpy array ECG signal or wfdb.Record object.
    :param fs: Sampling frequency.
    :param channels: List of channels to plot.
    :param duration: Duration of the signal to plot.
    """
    if not isinstance(record, (np.ndarray, wfdb.Record)):
        raise ValueError('record must be a numpy array or wfdb.Record object')
    if not isinstance(fs, int):
        raise ValueError('fs must be an integer')
    
    if isinstance(record, wfdb.Record):
        signal = record.p_signal
    else:
        signal = record

    time = np.linspace(0, duration, duration * fs)
    if channels == -1:
        try:
            channels = np.arange(signal.shape[1])
        except IndexError:
            channels = [0]
        num_channels = len(channels)
    else:
        num_channels = len(channels)
    
    sig_name = record.sig_name if isinstance(record, wfdb.Record) else None
    
    fig, ax = plt.subplots(num_channels, 1, figsize=(10, 4 * num_channels), sharex=True, sharey=True)
    if num_channels == 1:
        ax = [ax]
        signal = signal.reshape(-1, 1)
    fig.suptitle('ECG signal')
    fig.patch.set_facecolor("#ffe7e7")
    for i, channel in enumerate(channels):
        ax[i].plot(time, signal[:duration * fs, channel], color='blue')
        ax[i].set_ylabel('Amplitude')
        ax[i].set_title(f"CH{channel}{': ' + sig_name[channel] if sig_name is not None else ''}")
        ax[i].set_facecolor("#ffe7e7")
        ax[i].grid(color='red', linewidth=0.5)

    ax[-1].set_xlabel('Time [s]')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    return fig
