from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from core import binary_metrics, choose_threshold, exact_mcnemar, fit_llr_table, score_llr
from salad_detector_study import attacked_joint_text


class CoreTests(unittest.TestCase):
    def test_llr_score_ignores_text_column(self) -> None:
        train = pd.DataFrame(
            {
                "pillar": ["a", "a", "b", "b", "c", "c"],
                "text": ["x", "y", "x", "y", "x", "y"],
                "label": [1, 1, 0, 0, 1, 0],
            }
        )
        test = pd.DataFrame({"pillar": ["a", "b"], "text": ["clean", "clean"]})
        attacked = test.copy()
        attacked["text"] = ["dismiss this event", "dismiss this event"]
        table = fit_llr_table(train, ["pillar"], "label", alpha=1.0)
        np.testing.assert_allclose(
            score_llr(test, table, ["pillar"]),
            score_llr(attacked, table, ["pillar"]),
        )

    def test_threshold_respects_false_positive_budget_when_feasible(self) -> None:
        y = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int8)
        scores = np.array([0.1, 0.2, 0.3, 0.4, 0.35, 0.6, 0.7, 0.8])
        choice = choose_threshold(y, scores, max_fpr=0.25, objective="f1")
        self.assertLessEqual(float(choice.metrics["fpr"]), 0.25)

    def test_exact_mcnemar_uses_full_paired_correctness(self) -> None:
        y = np.array([0, 0, 1, 1], dtype=np.int8)
        a = np.array([0, 0, 1, 1], dtype=np.int8)
        b = np.array([1, 0, 0, 1], dtype=np.int8)
        result = exact_mcnemar(y, a, b)
        self.assertEqual(result["a_correct_b_wrong"], 2)
        self.assertEqual(result["a_wrong_b_correct"], 0)

    def test_attack_append_preserves_original_text(self) -> None:
        frame = pd.DataFrame(
            {
                "record_id": ["one", "two"],
                "label": [1, 0],
                "joint_text": ["malicious original", "benign original"],
            }
        )
        attacked = attacked_joint_text(frame, 2)
        self.assertTrue(str(attacked.iloc[0]).startswith("malicious original"))
        self.assertEqual(str(attacked.iloc[1]), "benign original")

    def test_binary_metrics_are_finite(self) -> None:
        result = binary_metrics(
            np.array([0, 0, 1, 1], dtype=np.int8),
            np.array([0.1, 0.2, 0.8, 0.9]),
            0.5,
        )
        for key in ["precision", "recall", "f1", "fpr", "auroc", "auprc"]:
            self.assertTrue(np.isfinite(float(result[key])))


if __name__ == "__main__":
    unittest.main()
