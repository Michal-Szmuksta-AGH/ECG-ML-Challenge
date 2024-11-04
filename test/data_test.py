import os
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src import dataset

TEST_DIR = 'tmp'
TEST_DATASET_DIR = 'mit-bih-arrhythmia-database'


class TestdatasetDownload(unittest.TestCase):
    """
    Test dataset download functions.
    """

    def test_download_dataset(self):
        """
        Test download_wfdb_dataset function.
        """
        dataset.download_wfdb_dataset('mitdb', os.path.join(TEST_DIR, TEST_DATASET_DIR))
        self.assertTrue(os.path.exists(os.path.join(TEST_DIR, TEST_DATASET_DIR)))
    
    def test_download_dataset_name_type_error(self):
        """
        Test download_wfdb_dataset function with error.
        """
        with self.assertRaises(ValueError):
            dataset.download_wfdb_dataset(123, os.path.join(TEST_DIR, TEST_DATASET_DIR))
    
    def test_download_dataset_dir_type_error(self):
        """
        Test download_wfdb_dataset function with error.
        """
        with self.assertRaises(ValueError):
            dataset.download_wfdb_dataset('mitdb', 123)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    unittest.main()
