import unittest

from scripts.generate_dataset import generate_lhs_samples


class DatasetSamplingTests(unittest.TestCase):
    def test_locations_do_not_reuse_identical_lhs_designs(self):
        first = generate_lhs_samples(20, seed=42)
        second = generate_lhs_samples(20, seed=43)
        self.assertFalse(first.equals(second))

    def test_single_lhs_has_unique_feature_vectors(self):
        samples = generate_lhs_samples(100, seed=42)
        self.assertFalse(samples.duplicated().any())


if __name__ == "__main__":
    unittest.main()
