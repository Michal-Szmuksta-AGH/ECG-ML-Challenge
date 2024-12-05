import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import wfdb.processing
from sklearn.preprocessing import MinMaxScaler

from signal_reader import SignalReader
from config import TRAINED_CHUNK_SIZE, TRAINED_FS, MODEL_FILE

class GPTMultiScaleConvGRUModel(nn.Module):
    def __init__(self) -> None:
        """
        Initialize the GPTMultiScaleConvGRUModel.
        """
        super(GPTMultiScaleConvGRUModel, self).__init__()

        # First scale
        self.conv1_1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.bn1_1 = nn.BatchNorm1d(16)
        self.conv1_2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.bn1_2 = nn.BatchNorm1d(32)
        self.conv1_3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn1_3 = nn.BatchNorm1d(64)
        self.shortcut1 = nn.Conv1d(1, 64, kernel_size=1, padding=0)

        # Second scale
        self.conv2_1 = nn.Conv1d(1, 16, kernel_size=7, padding=3)
        self.bn2_1 = nn.BatchNorm1d(16)
        self.conv2_2 = nn.Conv1d(16, 32, kernel_size=7, padding=3)
        self.bn2_2 = nn.BatchNorm1d(32)
        self.conv2_3 = nn.Conv1d(32, 64, kernel_size=7, padding=3)
        self.bn2_3 = nn.BatchNorm1d(64)
        self.shortcut2 = nn.Conv1d(1, 64, kernel_size=1, padding=0)

        # Third scale
        self.conv3_1 = nn.Conv1d(1, 16, kernel_size=15, padding=7)
        self.bn3_1 = nn.BatchNorm1d(16)
        self.conv3_2 = nn.Conv1d(16, 32, kernel_size=15, padding=7)
        self.bn3_2 = nn.BatchNorm1d(32)
        self.conv3_3 = nn.Conv1d(32, 64, kernel_size=15, padding=7)
        self.bn3_3 = nn.BatchNorm1d(64)
        self.shortcut3 = nn.Conv1d(1, 64, kernel_size=1, padding=0)

        # Layer normalization
        self.norm1 = nn.LayerNorm(192)  # Adjusted for increased channels

        # GRU layer
        self.gru = nn.GRU(
            input_size=192, hidden_size=128, num_layers=3, batch_first=True, bidirectional=True
        )

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 2, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the GPTMultiScaleConvGRUModel.

        :param x: Input tensor of shape [batch_size, seq_length].
        :return: Output tensor of shape [batch_size, seq_length].
        """
        x = x.unsqueeze(1)

        # First scale with skip connection
        x1 = F.relu(self.bn1_1(self.conv1_1(x)))
        x1 = F.relu(self.bn1_2(self.conv1_2(x1)))
        x1 = F.relu(self.bn1_3(self.conv1_3(x1) + self.shortcut1(x)))  # Skip connection
        x1 = F.adaptive_max_pool1d(x1, output_size=x.size(-1))

        # Second scale with skip connection
        x2 = F.relu(self.bn2_1(self.conv2_1(x)))
        x2 = F.relu(self.bn2_2(self.conv2_2(x2)))
        x2 = F.relu(self.bn2_3(self.conv2_3(x2) + self.shortcut2(x)))  # Skip connection
        x2 = F.adaptive_max_pool1d(x2, output_size=x.size(-1))

        # Third scale with skip connection
        x3 = F.relu(self.bn3_1(self.conv3_1(x)))
        x3 = F.relu(self.bn3_2(self.conv3_2(x3)))
        x3 = F.relu(self.bn3_3(self.conv3_3(x3) + self.shortcut3(x)))  # Skip connection
        x3 = F.adaptive_max_pool1d(x3, output_size=x.size(-1))

        # Concatenate features from all scales
        x = torch.cat((x1, x2, x3), dim=1)  # Shape: [batch_size, 192, seq_length]

        # Layer normalization
        x = x.transpose(1, 2)  # Shape: [batch_size, seq_length, 192]
        x = self.norm1(x)

        # GRU
        x, _ = self.gru(x)  # Output shape: [batch_size, seq_length, 256]

        # Fully connected layers
        x = F.relu(self.fc1(x))  # Shape: [batch_size, seq_length, 64]
        x = self.fc2(x)  # Shape: [batch_size, seq_length, 1]
        # x = torch.sigmoid(x)

        return x.squeeze(-1)  # Output shape: [batch_size, seq_length]


class RecordEvaluator:
    def __init__(self, dest_dir):
        self._dest_dir = dest_dir
        self._model = GPTMultiScaleConvGRUModel()
        state_dict = torch.load(f"./{MODEL_FILE}", weights_only=True, map_location="cpu")
        prefix = "_orig_mod."
        state_dict = {k[len(prefix) :] if k.startswith(prefix) else k: v for k, v in state_dict.items()}
        self._model.load_state_dict(state_dict)
        self._model.eval()
        self._scaler = MinMaxScaler(feature_range=(-1, 1))

    def evaluate(self, signal_reader):
        signal = signal_reader.read_signal().astype(np.float32)
        fs = signal_reader.read_fs()
        signal = signal.T
        ch, orig_sig_len = signal.shape
        
        # Resampling
        if fs != TRAINED_FS:
            resampled_signals = []
            for i in range(ch):
                s, _ = wfdb.processing.resample_sig(signal[i, ...], fs, TRAINED_FS)
                resampled_signals.append(s)
            signal = np.array(resampled_signals, dtype=np.float32)
            del resampled_signals
        else:
            signal = signal
        
        # Spliting into chunks
        _, resampled_sig_len = signal.shape
        if resampled_sig_len != TRAINED_CHUNK_SIZE:
            num_chunks = int(np.ceil(resampled_sig_len / TRAINED_CHUNK_SIZE))
            chunks = []
            
            for i in range(num_chunks):
                start_idx = i * TRAINED_CHUNK_SIZE
                end_idx = min((i + 1) * TRAINED_CHUNK_SIZE, resampled_sig_len)
                chunk = signal[:, start_idx:end_idx]
                
                # Padding
                if chunk.shape[1] < TRAINED_CHUNK_SIZE:
                    padding = np.zeros((ch, TRAINED_CHUNK_SIZE - chunk.shape[1]), dtype=np.float32)
                    chunk = np.concatenate((chunk, padding), axis=1, dtype=np.float32)
                chunks.append(chunk)
        else:
            chunks = [signal]
        del signal

        # Model forward pass
        preds = []
        for chunk in chunks:

            scaled_chunk = np.zeros_like(chunk, dtype=np.float32)
            for i in range(ch):
                scaled_chunk[i] = self._scaler.fit_transform(chunk[i].reshape(-1, 1)).ravel()

            del chunk
            scaled_chunk = torch.tensor(scaled_chunk, dtype=torch.float32)
            with torch.no_grad():
                pred = self._model(scaled_chunk)
                pred = torch.sigmoid(pred).numpy()
                preds.append(pred)

        preds = np.concatenate(preds, axis=1, dtype=np.float32)[:resampled_sig_len]
        
        # Resampling back to original fs
        if fs != TRAINED_FS:
            final_signal = []
            t_fs = int((orig_sig_len / preds.shape[1]) * TRAINED_FS + 1)
            for i in range(ch):
                s, _ = wfdb.processing.resample_sig(preds[i, ...], TRAINED_FS, t_fs)
                final_signal.append(s)
            preds = np.array(final_signal)
            del final_signal
        else:
            preds = preds

        preds = np.mean(preds, axis=0, dtype=np.float32)
        preds = preds > 0.5
        preds = preds[:orig_sig_len]

        code = signal_reader.get_code()
        np.save(os.path.join(self._dest_dir, f"{code}"), preds)


if __name__ == "__main__":
    record_eval = RecordEvaluator('./')
    signal_reader = SignalReader("./val_db/6.csv")
    record_eval.evaluate(signal_reader)
