"""Evidence-integrity tests, including deliberate corruptions."""
import copy
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import check_release as audit


class ReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = audit.rows(ROOT / "data/processed/aims_clean_features.csv")
        cls.predictions = audit.rows(ROOT / "results/reference/final_predictions.csv")

    def test_dataset_and_split(self):
        self.assertEqual(len(audit.validate_dataset(self.data)), 668)

    def test_predictions_and_metrics(self):
        result = audit.validate_tables(ROOT, self.data, self.predictions)
        self.assertAlmostEqual(result["EL"]["historical_test_MAE"], 13.171299790, places=7)

    def test_changed_target_rejected(self):
        records = copy.deepcopy(self.predictions)
        records[0]["target"] = str(float(records[0]["target"]) + 1)
        with self.assertRaisesRegex(ValueError, "Prediction target"):
            audit.validate_predictions(self.data, records)

    def test_duplicate_prediction_rejected(self):
        with self.assertRaisesRegex(ValueError, "population"):
            audit.validate_predictions(self.data, self.predictions + [self.predictions[0]])

    def test_changed_split_rejected(self):
        data = copy.deepcopy(self.data)
        data[0]["Split"] = "test"
        with self.assertRaisesRegex(ValueError, "split"):
            audit.validate_dataset(data)

    def test_provenance_hashes(self):
        self.assertGreater(audit.validate_hashes(ROOT), 50)

    def test_public_tree_and_notebooks(self):
        self.assertGreaterEqual(audit.validate_public_tree(ROOT), 7)

    def test_quantile(self):
        self.assertAlmostEqual(audit.quantile([0, 10, 20], .9), 18)


if __name__ == "__main__":
    unittest.main()
