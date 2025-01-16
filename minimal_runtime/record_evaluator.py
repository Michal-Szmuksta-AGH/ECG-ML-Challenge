import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import wfdb
from wfdb import processing
from sklearn.preprocessing import MinMaxScaler

from signal_reader import SignalReader
from config import TRAINED_CHUNK_SIZE, TRAINED_FS, MODEL_FILE


class MultiScaleConvLSTMModelv2(nn.Module):
    def __init__(self) -> None:
        """
        Initialize the GPTMultiScaleConvGRUModel.
        """
        super(MultiScaleConvLSTMModelv2, self).__init__()

        # First scale
        self.conv1_1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.bn1_1 = nn.BatchNorm1d(16)
        self.conv1_2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.bn1_2 = nn.BatchNorm1d(32)
        self.conv1_3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn1_3 = nn.BatchNorm1d(64)
        self.shortcut1 = nn.Conv1d(1, 64, kernel_size=1, padding=0)
        self.dropout1 = nn.Dropout(p=0.3)

        # Second scale
        self.conv2_1 = nn.Conv1d(1, 16, kernel_size=5, padding=2)
        self.bn2_1 = nn.BatchNorm1d(16)
        self.conv2_2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.bn2_2 = nn.BatchNorm1d(32)
        self.conv2_3 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2_3 = nn.BatchNorm1d(64)
        self.shortcut2 = nn.Conv1d(1, 64, kernel_size=1, padding=0)
        self.dropout2 = nn.Dropout(p=0.3)

        # Third scale
        self.conv3_1 = nn.Conv1d(1, 16, kernel_size=7, padding=3)
        self.bn3_1 = nn.BatchNorm1d(16)
        self.conv3_2 = nn.Conv1d(16, 32, kernel_size=7, padding=3)
        self.bn3_2 = nn.BatchNorm1d(32)
        self.conv3_3 = nn.Conv1d(32, 64, kernel_size=7, padding=3)
        self.bn3_3 = nn.BatchNorm1d(64)
        self.shortcut3 = nn.Conv1d(1, 64, kernel_size=1, padding=0)
        self.dropout3 = nn.Dropout(p=0.3)

        # Layer normalization
        self.norm1 = nn.LayerNorm(192)  # Adjusted for increased channels

        # GRU layer
        self.gru = nn.LSTM(
            input_size=192, hidden_size=512, num_layers=3, batch_first=True, bidirectional=True
        )
        self.dropout_gru = nn.Dropout(p=0.4)

        # Fully connected layers
        self.fc1 = nn.Linear(512 * 2, 64)
        self.dropout_fc = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MultiScaleConvGRUModel.

        :param x: Input tensor of shape [batch_size, seq_length].
        :return: Output tensor of shape [batch_size, seq_length].
        """
        x = x.permute(0, 2, 1)

        # First scale with skip connection
        x1 = F.relu(self.bn1_1(self.conv1_1(x)))
        x1 = F.relu(self.bn1_2(self.conv1_2(x1)))
        x1 = F.relu(self.bn1_3(self.conv1_3(x1) + self.shortcut1(x)))  # Skip connection
        x1 = self.dropout1(x1)
        x1 = F.adaptive_max_pool1d(x1, output_size=x.size(-1))

        # Second scale with skip connection
        x2 = F.relu(self.bn2_1(self.conv2_1(x)))
        x2 = F.relu(self.bn2_2(self.conv2_2(x2)))
        x2 = F.relu(self.bn2_3(self.conv2_3(x2) + self.shortcut2(x)))  # Skip connection
        x2 = self.dropout2(x2)
        x2 = F.adaptive_max_pool1d(x2, output_size=x.size(-1))

        # Third scale with skip connection
        x3 = F.relu(self.bn3_1(self.conv3_1(x)))
        x3 = F.relu(self.bn3_2(self.conv3_2(x3)))
        x3 = F.relu(self.bn3_3(self.conv3_3(x3) + self.shortcut3(x)))  # Skip connection
        x3 = self.dropout3(x3)
        x3 = F.adaptive_max_pool1d(x3, output_size=x.size(-1))

        # Concatenate features from all scales
        x = torch.cat((x1, x2, x3), dim=1)  # Shape: [batch_size, 192, seq_length]

        # Layer normalization
        x = x.transpose(1, 2)  # Shape: [batch_size, seq_length, 192]
        x = self.norm1(x)

        # GRU
        x, _ = self.gru(x)  # Output shape: [batch_size, seq_length, 256]
        x = self.dropout_gru(x)

        # Fully connected layers
        x = F.relu(self.fc1(x))  # Shape: [batch_size, seq_length, 64]
        x = self.dropout_fc(x)
        x = self.fc2(x)  # Shape: [batch_size, seq_length, 1]
        x = torch.sigmoid(x)

        return x  # Output shape: [batch_size, seq_length, 1]


class RecordEvaluator:
    def __init__(self, dest_dir):
        self._dest_dir = dest_dir
        self._model = MultiScaleConvLSTMModelv2()
        state_dict = torch.load(f"./{MODEL_FILE}", weights_only=True, map_location="cpu")
        prefix = "_orig_mod."
        state_dict = {
            k[len(prefix) :] if k.startswith(prefix) else k: v for k, v in state_dict.items()
        }
        self._model.load_state_dict(state_dict)
        self._model.eval()

    def evaluate(self, signal_reader):
        signal = signal_reader.read_signal().astype(np.float32)
        fs = signal_reader.read_fs()
        signal = signal.T

        # Resampling
        if fs != TRAINED_FS:
            signal, _ = wfdb.processing.resample_sig(signal[0, ...], fs, TRAINED_FS)
            signal = signal.astype(np.float32)
        else:
            signal = signal[0, ...]
            signal = signal.astype(np.float32)
        fs = TRAINED_FS

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

        input_rr_samples = TRAINED_CHUNK_SIZE
        batch_size = 64
        qrs_af_probabs = np.zeros(shape=(len(qrs_inds),), dtype=np.float32)
        qrs_af_overlap = np.zeros(shape=(len(qrs_inds),), dtype=np.float32)

        pred_step = 12

        batch = np.zeros(shape=(batch_size, input_rr_samples, 1), dtype=np.float32)
        batch_idx = 0
        rr_indices_history = []
        for rr_idx in range(0, rr.shape[0] - input_rr_samples, pred_step):
            snippet = rr[rr_idx : rr_idx + input_rr_samples]
            rr_indices_history.append([rr_idx, rr_idx + input_rr_samples])
            snippet = snippet[..., np.newaxis]
            batch[batch_idx] = snippet
            batch_idx += 1

            if batch_idx == batch_size:
                with torch.no_grad():
                    results = self._model(torch.from_numpy(batch).float()).numpy()
                for j in range(batch_idx):
                    rr_from, rr_to = rr_indices_history[j]
                    qrs_af_probabs[rr_from:rr_to] += results[j, :, 0]
                    qrs_af_overlap[rr_from:rr_to] += 1.0

                batch_idx = 0
                rr_indices_history = []

        if batch_idx > 0:
            with torch.no_grad():
                results = self._model(torch.from_numpy(batch).float()).numpy()
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

        for qrs_idx in range(len(rr)):
            pred[qrs_inds[qrs_idx] : qrs_inds[qrs_idx + 1]] = qrs_af_preds[qrs_idx]

        code = signal_reader.get_code()
        np.save(os.path.join(self._dest_dir, f"{code}"), pred)


if __name__ == "__main__":
    record_eval = RecordEvaluator("./")
    signal_reader = SignalReader("./val_db/6.csv")
    record_eval.evaluate(signal_reader)
