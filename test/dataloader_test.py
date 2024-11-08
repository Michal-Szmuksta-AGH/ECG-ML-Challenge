import os
import sys
import unittest
import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utilities.dataloader import ECGDataset, ECGDataLoader

class TestECGDataset(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_dir = 'test_data'
        os.makedirs(cls.test_dir, exist_ok=True)
        for i in range(5):
            np.savez(os.path.join(cls.test_dir, f'test_file_{i}.npz'), x=np.random.rand(5000, 1), y=np.random.randint(0, 2, size=(5000, 1)))

    @classmethod
    def tearDownClass(cls):
        for f in os.listdir(cls.test_dir):
            os.remove(os.path.join(cls.test_dir, f))
        os.rmdir(cls.test_dir)

    def test_len(self):
        dataset = ECGDataset(self.test_dir)
        self.assertEqual(len(dataset), 5)

    def test_getitem(self):
        dataset = ECGDataset(self.test_dir)
        x, y = dataset[0]
        self.assertIsInstance(x, torch.Tensor)
        self.assertIsInstance(y, torch.Tensor)
        self.assertEqual(x.dtype, torch.float32)
        self.assertEqual(y.dtype, torch.uint8)
        self.assertEqual(x.shape, (5000, 1))
        self.assertEqual(y.shape, (5000, 1))

    def test_various_num_channels(self):
        test_dir = 'test_data_diff_channels'
        os.makedirs(test_dir, exist_ok=True)
        try:
            fixed_size = 5000
            channels = [1, 2, 3, 4, 5]
            for i, channel in enumerate(channels):
                np.savez(os.path.join(test_dir, f'test_file_{i}.npz'), x=np.random.rand(fixed_size, channel), y=np.random.randint(0, 2, size=(fixed_size, 1)))
            dataset = ECGDataset(test_dir)
            for i, channel in enumerate(channels):
                x, y = dataset[i]
                expected = np.load(os.path.join(test_dir, dataset.files[i]))
                x_exp, y_exp = expected['x'], expected['y']
                self.assertIsInstance(x, torch.Tensor)
                self.assertIsInstance(y, torch.Tensor)
                self.assertEqual(x.dtype, torch.float32)
                self.assertEqual(y.dtype, torch.uint8)
                self.assertEqual(x.shape, x_exp.shape)
                self.assertEqual(y.shape, y_exp.shape)
        finally:
            for f in os.listdir(test_dir):
                os.remove(os.path.join(test_dir, f))
            os.rmdir(test_dir)

    def test_various_lengths(self):
        test_dir = 'test_data_diff_lengths'
        os.makedirs(test_dir, exist_ok=True)
        try:
            sizes = [1000, 2000, 3000, 4000, 5000]
            fixed_channel = 1
            for i, size in enumerate(sizes):
                np.savez(os.path.join(test_dir, f'test_file_{i}.npz'), x=np.random.rand(size, fixed_channel), y=np.random.randint(0, 2, size=(size, 1)))
            dataset = ECGDataset(test_dir)
            for i, size in enumerate(sizes):
                x, y = dataset[i]
                expected = np.load(os.path.join(test_dir, dataset.files[i]))
                x_exp, y_exp = expected['x'], expected['y']
                self.assertIsInstance(x, torch.Tensor)
                self.assertIsInstance(y, torch.Tensor)
                self.assertEqual(x.dtype, torch.float32)
                self.assertEqual(y.dtype, torch.uint8)
                self.assertEqual(x.shape, x_exp.shape)
                self.assertEqual(y.shape, y_exp.shape)
        finally:
            for f in os.listdir(test_dir):
                os.remove(os.path.join(test_dir, f))
            os.rmdir(test_dir)

class TestECGDataLoader(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_dir = 'test_data_loader'
        os.makedirs(cls.test_dir, exist_ok=True)
        for i in range(10):
            np.savez(os.path.join(cls.test_dir, f'test_file_{i}.npz'), x=np.random.rand(5000, 1), y=np.random.randint(0, 2, size=(5000, 1)))

    @classmethod
    def tearDownClass(cls):
        for f in os.listdir(cls.test_dir):
            os.remove(os.path.join(cls.test_dir, f))
        os.rmdir(cls.test_dir)

    def test_dataloader_len(self):
        dataloader = ECGDataLoader(self.test_dir, batch_size=2)
        self.assertEqual(len(dataloader), 5)

    def test_dataloader_batch(self):
        dataloader = ECGDataLoader(self.test_dir, batch_size=2)
        for batch in dataloader:
            x, y = batch
            self.assertIsInstance(x, torch.Tensor)
            self.assertIsInstance(y, torch.Tensor)
            self.assertEqual(x.dtype, torch.float32)
            self.assertEqual(y.dtype, torch.uint8)
            self.assertEqual(x.shape[0], 2)
            self.assertEqual(y.shape[0], 2)
            break  # Only test the first batch

if __name__ == '__main__':
    unittest.main()
