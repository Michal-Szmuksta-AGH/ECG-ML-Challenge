import os
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset


class ECGDataset(Dataset):
    """
    Dataset class for ECG data
    """

    def __init__(
        self,
        directory: str,
        input_dtype: torch.dtype = torch.float32,
        output_dtype: torch.dtype = torch.uint8,
    ):
        """
        ECGDataset constructor
        :param directory: Directory containing the ECG data files
        :param input_dtype: Data type of the ECG signal
        :param output_dtype: Data type of the expected annotation vector
        """
        self.directory: str = directory
        self.input_dtype: torch.dtype = input_dtype
        self.output_dtype: torch.dtype = output_dtype
        self.files: List[str] = [f for f in os.listdir(directory) if f.endswith(".npz")]
        if not self.files:
            raise ValueError(f"No .npz files found in directory: {directory}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.directory, self.files[idx])
        data = np.load(file_path)
        x = data["x"]
        y = data["y"]
        x = torch.tensor(x, dtype=self.input_dtype)
        y = torch.tensor(y, dtype=self.output_dtype)
        return x, y
