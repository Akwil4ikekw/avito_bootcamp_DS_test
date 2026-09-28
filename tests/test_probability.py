"""Check the known PaddleX batch-label edge case without importing Paddle."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("orientation_predict", ROOT / "src/predict.py")
PREDICT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREDICT)


class ProbabilityTests(unittest.TestCase):
    def test_uses_per_image_labels_not_batch_matrix(self):
        result = {"class_ids": np.array([[0, 1], [1, 0]]),
                  "scores": [0.97, 0.03], "label_names": ["180_degree", "0_degree"]}
        self.assertAlmostEqual(PREDICT.probability_180(result), 0.97)

    def test_nested_result_and_label_order(self):
        result = {"res": {"scores": [0.8, 0.2], "label_names": ["0_degree", "180_degree"]}}
        self.assertAlmostEqual(PREDICT.probability_180(result), 0.2)

    def test_rejects_invalid_probabilities(self):
        for scores in ([float("nan"), 0.2], [1.1, -0.1], [0.7, 0.7]):
            with self.subTest(scores=scores), self.assertRaises(ValueError):
                PREDICT.probability_180({"scores": scores, "label_names": ["0_degree", "180_degree"]})

    def test_top_one_is_not_silently_used(self):
        with self.assertRaises(ValueError):
            PREDICT.probability_180({"scores": [0.9], "label_names": ["180_degree"]})


if __name__ == "__main__":
    unittest.main()
