import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import wfdb
from wfdb import processing

from signal_reader import SignalReader
from config import TRAINED_CHUNK_SIZE, TRAINED_FS, MODEL_FILE


class ConvBlock(nn.Module):
    def __init__(self, kernel_size):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=kernel_size, padding=kernel_size // 2)
        self.dropout1 = nn.Dropout(p=0.4)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=kernel_size, padding=kernel_size // 2)
        self.dropout2 = nn.Dropout(p=0.4)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=kernel_size, padding=kernel_size // 2)
        self.dropout3 = nn.Dropout(p=0.4)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=kernel_size, padding=kernel_size // 2)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = self.dropout1(x)
        x = F.leaky_relu(self.conv2(x))
        x = self.dropout2(x)
        x = F.leaky_relu(self.conv3(x))
        x = self.dropout3(x)
        x = F.leaky_relu(self.conv4(x))

        return x


class LastChance(nn.Module):
    def __init__(self):
        super(LastChance, self).__init__()
        self.skipconv = nn.Conv1d(1, 128, kernel_size=1)
        self.skip_dropout = nn.Dropout(p=0.4)
        self.convblock1 = ConvBlock(5)
        self.convblock2 = ConvBlock(7)
        self.convblock3 = ConvBlock(9)
        self.LSTM = nn.LSTM(128 * 4, 256, num_layers=4, batch_first=True, bidirectional=True)
        self.dropout1 = nn.Dropout(p=0.4)
        self.dropout2 = nn.Dropout(p=0.4)
        self.dense1 = nn.Linear(512, 256)
        self.dense2 = nn.Linear(256, 32)

    def forward(self, x):
        # x = x.unsqueeze(1)
        skip = F.leaky_relu(self.skipconv(x))
        skip = self.skip_dropout(skip)
        x1 = self.convblock1(x)
        x2 = self.convblock2(x)
        x3 = self.convblock3(x)
        x = torch.cat([skip, x1, x2, x3], dim=1)
        x = x.permute(0, 2, 1)
        x, _ = self.LSTM(x)
        x = self.dropout1(x)
        x = x[:, -1, :]
        x = self.dense1(x)
        x = self.dropout2(x)
        x = self.dense2(x)

        return x


class RecordEvaluator:
    def __init__(self, dest_dir):
        self._dest_dir = dest_dir
        self._model = LastChance()
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

        # Resampling
        if fs != TRAINED_FS:
            signal, _ = wfdb.processing.resample_sig(signal[..., 0], fs, TRAINED_FS)
            signal = signal.astype(np.float32)
        else:
            signal = signal[..., 0]
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
            qrs_inds, fs=fs, min_rr=None, max_rr=None, qrs_units="samples", rr_units="seconds"
        )

        input_rr_samples = TRAINED_CHUNK_SIZE
        batch_size = 64
        qrs_af_probabs = np.zeros(shape=(len(qrs_inds),), dtype=np.float32)
        qrs_af_overlap = np.zeros(shape=(len(qrs_inds),), dtype=np.float32)

        pred_step = 8

        batch = np.zeros(shape=(batch_size, 1, input_rr_samples), dtype=np.float32)
        batch_idx = 0
        rr_indices_history = []
        for rr_idx in range(0, rr.shape[0] - input_rr_samples, pred_step):
            snippet = rr[rr_idx : rr_idx + input_rr_samples]
            rr_indices_history.append([rr_idx, rr_idx + input_rr_samples])
            snippet = snippet[np.newaxis, :]
            batch[batch_idx] = snippet
            batch_idx += 1

            if batch_idx == batch_size:
                with torch.no_grad():
                    results = self._model(torch.from_numpy(batch).float())
                    results = torch.sigmoid(results).numpy()
                for j in range(batch_idx):
                    rr_from, rr_to = rr_indices_history[j]
                    qrs_af_probabs[rr_from:rr_to] += results[j, :]
                    qrs_af_overlap[rr_from:rr_to] += 1.0

                batch_idx = 0
                rr_indices_history = []

        if batch_idx > 0:
            batch = batch[:batch_idx]
            with torch.no_grad():
                results = self._model(torch.from_numpy(batch).float())
                results = torch.sigmoid(results).numpy()
            for j in range(batch_idx):
                rr_from, rr_to = rr_indices_history[j]
                qrs_af_probabs[rr_from:rr_to] += results[j, :]
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
