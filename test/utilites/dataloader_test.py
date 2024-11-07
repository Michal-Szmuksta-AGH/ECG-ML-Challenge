import os
import unittest
import numpy as np
import torch
from src.utilities.dataloader import ECGDataset

class TestECGDataset(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_dir = 'test_data'
        os.makedirs(cls.test_dir, exist_ok=True)
        for i in range(5):
            np.savez(os.path.join(cls.test_dir, f'test_file_{i}.npz'), x=np.random.rand(5000), y=np.random.randint(0, 2, size=1))

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
        self.assertEqual(x.shape, (5000,))
        self.assertEqual(y.shape, (1,))

if __name__ == '__main__':
    unittest.main()
