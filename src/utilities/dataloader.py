import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List

class ECGDataset(Dataset):
    """
    Dataset class for ECG data
    """

    def __init__(self, directory: str, input_dtype: torch.dtype = torch.float32, output_dtype: torch.dtype = torch.uint8):
        """
        ECGDataset constructor
        :param directory: Directory containing the ECG data files
        :param input_dtype: Data type of the ECG signal
        :param output_dtype: Data type of the expected annotation vector
        """
        self.directory: str = directory
        self.input_dtype: torch.dtype = input_dtype
        self.output_dtype: torch.dtype = output_dtype
        self.files: List[str] = [f for f in os.listdir(directory) if f.endswith('.npz')]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.directory, self.files[idx])
        data = np.load(file_path)
        x = data['x']
        y = data['y']
        x = torch.tensor(x, dtype=self.input_dtype)
        y = torch.tensor(y, dtype=self.output_dtype)
        return x, y

# def create_dataloader(directory, batch_size=32, shuffle=True, num_workers=4):
#     dataset = ECGDataset(directory)
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
#     return dataloader

class ECGDataLoader(DataLoader):
    """
    DataLoader class for ECG data
    """

    def __init__(
            self, directory: str,
            input_dtype: torch.dtype = torch.float32,
            output_dtype: torch.dtype = torch.uint8,
            batch_size: int = 32,
            shuffle: bool = True, 
            num_workers: int = 4):
        """
        ECGDataLoader constructor
        :param directory: Directory containing the ECG data files
        :param input_dtype: Data type of the ECG signal
        :param output_dtype: Data type of the expected annotation vector
        :param batch_size: Batch size
        :param shuffle: Whether to shuffle the data
        :param num_workers: Number of workers for data loading
        """
        dataset = ECGDataset(
            directory,
            input_dtype=input_dtype,
            output_dtype=output_dtype
            )
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers
            )