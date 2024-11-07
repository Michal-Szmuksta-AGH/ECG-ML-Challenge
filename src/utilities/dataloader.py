import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List

class ECGDataset(Dataset):
    """
    Dataset class for ECG data
    """

    def __init__(self, directory: str, signal_dtype: torch.dtype = torch.float32):
        """
        ECGDataset constructor
        :param directory: Directory containing the ECG data files
        :param signal_dtype: Data type of the ECG signal
        """
        self.directory: str = directory
        self.files: List[str] = [f for f in os.listdir(directory) if f.endswith('.npz')]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.directory, self.files[idx])
        data = np.load(file_path)
        x = data['x']
        y = data['y']
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.uint8)
        return x, y

# def create_dataloader(directory, batch_size=32, shuffle=True, num_workers=4):
#     dataset = ECGDataset(directory)
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
#     return dataloader