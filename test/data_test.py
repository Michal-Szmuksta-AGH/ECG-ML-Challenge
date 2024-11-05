import os
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src import dataset

TEST_DIR = "tmp"
TEST_DATASET_DIR = os.abspath(os.path.join(TEST_DIR, "mit-bih-arrhythmia-database"))


class TestDatasetDownload(unittest.TestCase):
    """
    Test dataset download functions.
    """

    def test_download_dataset(self):
        """
        Test download_wfdb_dataset function.
        """
        dataset.download_wfdb_dataset("mitdb", TEST_DATASET_DIR)
        self.assertTrue(os.path.exists(TEST_DATASET_DIR))

    def test_download_dataset_name_type_error(self):
        """
        Test download_wfdb_dataset function with error.
        """
        with self.assertRaises(ValueError):
            dataset.download_wfdb_dataset(123, TEST_DATASET_DIR)

    def test_download_dataset_dir_type_error(self):
        """
        Test download_wfdb_dataset function with error.
        """
        with self.assertRaises(ValueError):
            dataset.download_wfdb_dataset("mitdb", 123)


class TestDatasetAnn2vec(unittest.TestCase):
    """
    Test annotation conversion functions.
    """

    def test_ann2vec(self):
        """
        Test ann2vec function.
        """
        ann = dataset.wfdb.rdann(os.path.join(TEST_DATASET_DIR, "100"), "atr")
        ann_vec = dataset.ann2vec(ann, 360, 650000)
        self.assertEqual(ann_vec.shape, (650000,))

    def test_ann2vec_annotation_type_error(self):
        """
        Test ann2vec function with error.
        """
        with self.assertRaises(ValueError):
            dataset.ann2vec(123, 360, 650000)

    def test_ann2vec_fs_type_error(self):
        """
        Test ann2vec function with error.
        """
        ann = dataset.wfdb.rdann(os.path.join(TEST_DATASET_DIR, "100"), "atr")
        with self.assertRaises(ValueError):
            dataset.ann2vec(ann, "360", 650000)

    def test_ann2vec_samples_type_error(self):
        """
        Test ann2vec function with error.
        """
        ann = dataset.wfdb.rdann(os.path.join(TEST_DATASET_DIR, "100"), "atr")
        with self.assertRaises(ValueError):
            dataset.ann2vec(ann, 360, "650000")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    unittest.main()
