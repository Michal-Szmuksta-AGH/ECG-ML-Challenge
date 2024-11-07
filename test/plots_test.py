import os
import sys
import unittest
import wfdb
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src import plot_ecg

TEST_DIR = "tmp"
TEST_DATASET_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), TEST_DIR, "mit-bih-arrhythmia-database")
)


class TestPlots(unittest.TestCase):
    """
    Test plots functions.
    """

    def test_record_type_error(self):
        """
        Test record type error.
        """
        with self.assertRaises(ValueError):
            plot_ecg("record", 360)

    def test_fs_type_error(self):
        """
        Test fs type error.
        """
        with self.assertRaises(ValueError):
            plot_ecg([1, 2, 3], "fs")

    def test_channels_type_error(self):
        """
        Test channels type error.
        """
        with self.assertRaises(ValueError):
            plot_ecg([1, 2, 3], 360, channels="channels")

    def test_plot_ecg(self):
        """
        Test plot_ecg function.
        """
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        record = wfdb.rdrecord(os.path.join(TEST_DATASET_DIR, "103"))
        fig = plot_ecg(record.p_signal, record.fs, t_start=10, t_end=25)
        self.assertIsInstance(fig, plt.Figure)
        plt.savefig("tmp/test_plot_ecg.png")

    def test_plot_ecg_annotation(self):
        """
        Test plot_ecg function with annotation.
        """
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        record = wfdb.rdrecord(os.path.join(TEST_DATASET_DIR, "103"))
        annotation = wfdb.rdann(os.path.join(TEST_DATASET_DIR, "103"), "atr")
        fig = plot_ecg(record.p_signal, record.fs, t_start=10, t_end=25, annotation=annotation)
        self.assertIsInstance(fig, plt.Figure)
        plt.savefig("tmp/test_plot_ecg_annotation.png")

    def test_plot_ecg_no_finish_time(self):
        """
        Test plot_ecg function without finish time.
        """
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        record = wfdb.rdrecord(os.path.join(TEST_DATASET_DIR, "103"))
        fig = plot_ecg(record.p_signal, record.fs, t_start=10)
        self.assertIsInstance(fig, plt.Figure)
        plt.savefig("tmp/test_plot_ecg_no_finish_time.png")


if __name__ == "__main__":

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    if not os.path.exists(TEST_DIR):
        os.makedirs(TEST_DIR)

    if not os.path.exists(TEST_DATASET_DIR) or len(os.listdir(TEST_DATASET_DIR)) == 0:
        print("Downloading database...")
        wfdb.dl_database("mitdb", dl_dir=TEST_DATASET_DIR)
    else:
        print("Database already downloaded!")

    unittest.main()
