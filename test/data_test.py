import os
import sys
import unittest
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.dataset import processing

TEST_DIR = "tmp"
TEST_DATASET_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), TEST_DIR, "numpy-ecg-dataset")
)



class TestDatasetDownload(unittest.TestCase):
    """
    Test dataset download functions.
    """

    def test_download_dataset(self):
        """
        Test download_wfdb_dataset function.
        """
        processing.download_wfdb_dataset("mitdb", TEST_DATASET_DIR)
        self.assertTrue(os.path.exists(TEST_DATASET_DIR))

    def test_download_dataset_name_type_error(self):
        """
        Test download_wfdb_dataset function with error.
        """
        with self.assertRaises(ValueError):
            processing.download_wfdb_dataset(123, TEST_DATASET_DIR)

    def test_download_dataset_dir_type_error(self):
        """
        Test download_wfdb_dataset function with error.
        """
        with self.assertRaises(ValueError):
            processing.download_wfdb_dataset("mitdb", 123)


class TestDatasetSym2vec(unittest.TestCase):
    """
    Test annotation conversion functions.
    """

    def test_ann2vec(self):
        """
        Test ann2vec function.
        """
        processing.download_wfdb_dataset("mitdb", TEST_DATASET_DIR)
        ann = processing.wfdb.rdann(os.path.join(TEST_DATASET_DIR, "100"), "atr")
        ann_vec = processing.sym2vec(ann, 360, 650000)
        self.assertEqual(ann_vec.shape, (650000,))

    def test_ann2vec_annotation_type_error(self):
        """
        Test ann2vec function with error.
        """
        with self.assertRaises(ValueError):
            processing.sym2vec(123, 360, 650000)

    def test_ann2vec_fs_type_error(self):
        """
        Test ann2vec function with error.
        """
        processing.download_wfdb_dataset("mitdb", TEST_DATASET_DIR)
        ann = processing.wfdb.rdann(os.path.join(TEST_DATASET_DIR, "100"), "atr")
        with self.assertRaises(ValueError):
            processing.sym2vec(ann, "360", 650000)

    def test_ann2vec_samples_type_error(self):
        """
        Test ann2vec function with error.
        """
        processing.download_wfdb_dataset("mitdb", TEST_DATASET_DIR)
        ann = processing.wfdb.rdann(os.path.join(TEST_DATASET_DIR, "100"), "atr")
        with self.assertRaises(ValueError):
            processing.sym2vec(ann, 360, "650000")


class TestDatasetAux2vec(unittest.TestCase):
    """
    Test aux note conversion functions.
    """

    def test_aux2vec(self):
        """
        Test aux2vec function.
        """
        processing.download_wfdb_dataset("mitdb", TEST_DATASET_DIR)
        ann = processing.wfdb.rdann(os.path.join(TEST_DATASET_DIR, "100"), "atr")
        aux_vec = processing.aux2vec(ann, 360, 650000)
        self.assertEqual(aux_vec.shape, (650000,))

    def test_aux2vec_annotation_type_error(self):
        """
        Test aux2vec function with error.
        """
        with self.assertRaises(ValueError):
            processing.aux2vec(123, 360, 650000)

    def test_aux2vec_fs_type_error(self):
        """
        Test aux2vec function with error.
        """
        processing.download_wfdb_dataset("mitdb", TEST_DATASET_DIR)
        ann = processing.wfdb.rdann(os.path.join(TEST_DATASET_DIR, "100"), "atr")
        with self.assertRaises(ValueError):
            processing.aux2vec(ann, "360", 650000)

    def test_aux2vec_samples_type_error(self):
        """
        Test aux2vec function with error.
        """
        processing.download_wfdb_dataset("mitdb", TEST_DATASET_DIR)
        ann = processing.wfdb.rdann(os.path.join(TEST_DATASET_DIR, "100"), "atr")
        with self.assertRaises(ValueError):
            processing.aux2vec(ann, 360, "650000")


class TestDatasetSplitData(unittest.TestCase):
    """
    Test data splitting functions.
    """

    def test_split_data(self):
        """
        Test split_data function with valid input.
        """
        data = np.random.rand(1000)
        fs = 250
        chunk_size = 250
        chunks = processing.split_data(data, fs, chunk_size)
        self.assertEqual(chunks.shape, (4, 250))

    def test_split_data_type_error_data(self):
        """
        Test split_data function with data type error.
        """
        with self.assertRaises(ValueError):
            processing.split_data("not an array", 250, 250)

    def test_split_data_type_error_fs(self):
        """
        Test split_data function with fs type error.
        """
        data = np.random.rand(1000)
        with self.assertRaises(ValueError):
            processing.split_data(data, "250", 250)

    def test_split_data_type_error_chunk_size(self):
        """
        Test split_data function with chunk_size type error.
        """
        data = np.random.rand(1000)
        with self.assertRaises(ValueError):
            processing.split_data(data, 250, "250")

    def test_split_data_incomplete_chunk(self):
        """
        Test split_data function with data that doesn't fit perfectly into chunks.
        """
        data = np.random.rand(1020)
        fs = 250
        chunk_size = 250
        chunks = processing.split_data(data, fs, chunk_size)
        self.assertEqual(chunks.shape, (4, 250))


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    unittest.main()
