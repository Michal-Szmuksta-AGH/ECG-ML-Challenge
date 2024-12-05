import unittest

import os
import sys
import torch
import numpy as np
from unittest.mock import patch
from minimal_runtime.record_evaluator import RecordEvaluator

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src import config


class DummySignalReader:
    def read_signal(self) -> np.ndarray:
        return np.random.rand(8, 5000)  # 2 channels, 5000 samples

    def read_fs(self) -> float:
        return 250.0  # Example sampling frequency

    def get_code(self) -> str:
        return "2137"


class TestRecordEvaluator(unittest.TestCase):
    @patch("torch.load")
    def test_evaluate(self, mock_torch_load):
        mock_torch_load.side_effect = lambda path: torch.load(
            os.path.join(config.MINIMAL_RUNTIME_DIR, path)
        )

        evaluator = RecordEvaluator()
        signal_reader = DummySignalReader()

        with patch("numpy.save") as mock_save:
            evaluator.evaluate(signal_reader)
            mock_save.assert_called_once()
            args, kwargs = mock_save.call_args
            self.assertIn("2137.npy", args[0])


if __name__ == "__main__":
    unittest.main()
