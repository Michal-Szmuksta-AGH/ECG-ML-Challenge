import os
import torch
import numpy as np
import wfdb.processing

# from signal_reader import SignalReader
from .models import get_model
from .config import TRAINED_MODEL_TYPE, TRAINED_CHUNK_SIZE, TRAINED_FS, MODEL_FILE, DEST_DIR


class RecordEvaluator:
    def __init__(self):
        self._dest_dir = DEST_DIR
        self._model = get_model(TRAINED_MODEL_TYPE)
        self._model.load_state_dict(torch.load(f"./{MODEL_FILE}.pth"))
        self._model.eval()

    def evaluate(self, signal_reader):
        signal = signal_reader.read_signal()
        fs = signal_reader.read_fs()

        ch, sig_len = signal.shape

        if fs != TRAINED_FS:
            signals = []
            for i in range(ch):
                signal, _ = wfdb.processing.resample_sig(signal[i, ...], fs, TRAINED_FS)
                signals.append(signal)
            signal = np.array(signals)
        
        ch, sig_len = signal.shape

        if sig_len != TRAINED_CHUNK_SIZE:
            num_chunks = int(np.ceil(sig_len / TRAINED_CHUNK_SIZE))
            signal = np.array_split(signal, num_chunks, axis=1)

            last_chunk_len = signal[-1].shape[1]
            if last_chunk_len < TRAINED_CHUNK_SIZE:
                padding = np.zeros((ch, TRAINED_CHUNK_SIZE - last_chunk_len))
                signal[-1] = np.concatenate((signal[-1], padding), axis=1)
        
        if sig_len % TRAINED_CHUNK_SIZE != 0:
            padding = TRAINED_CHUNK_SIZE - (sig_len % TRAINED_CHUNK_SIZE)
            signal = np.pad(signal, ((0, 0), (0, padding)), "constant")

        num_chunks = sig_len // TRAINED_CHUNK_SIZE
        chunks = [signal[i * TRAINED_CHUNK_SIZE : (i + 1) * TRAINED_CHUNK_SIZE] for i in range(num_chunks)]

        preds = []
        for chunk in signal:
            chunk_tensor = torch.tensor(chunk).unsqueeze(0)
            with torch.no_grad():
                pred = self._model(chunk_tensor)
                pred = torch.mean(pred, dim=0).numpy()
                pred = np.round(pred)
                preds.append(pred)

        pred = np.concatenate(preds, axis=1)

        code = signal_reader.get_code()
        np.save(os.path.join(self._dest_dir, f"{code}"), pred)


if __name__ == "__main__":
    record_eval = RecordEvaluator()
    # signal_reader = SignalReader("./val_db/6.csv")
    signal_reader = None
    record_eval.evaluate(signal_reader)
