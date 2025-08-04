import unittest
import numpy as np
from main import euclidean, k_medoids, clara, dbscan

class TestClusteringMethods(unittest.TestCase):
    def setUp(self):
        # Simple dataset for testing
        self.X = np.array([
            [0, 0, 0],
            [0, 1, 0],
            [10, 10, 10],
            [10, 11, 10],
            [50, 50, 50]
        ])

    def test_euclidean(self):
        a = np.array([0, 0, 0])
        b = np.array([3, 4, 0])
        self.assertAlmostEqual(euclidean(a, b), 5.0)

    def test_k_medoids(self):
        labels, medoids = k_medoids(self.X, k=2, max_iter=10)
        self.assertEqual(len(labels), len(self.X))
        self.assertEqual(medoids.shape[0], 2)

    def test_clara(self):
        labels, medoids = clara(self.X, k=2, n_samples=3, n_iter=3)
        self.assertEqual(len(labels), len(self.X))
        self.assertEqual(medoids.shape[0], 2)

    def test_dbscan(self):
        labels = dbscan(self.X, eps=2, min_pts=1)
        self.assertEqual(len(labels), len(self.X))
        # At least one cluster should be found
        self.assertTrue(np.any(labels != -1))

if __name__ == '__main__':
    unittest.main()
