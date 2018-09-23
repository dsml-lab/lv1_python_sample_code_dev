import unittest
import numpy as np
from democ.distance import calc_distance
import coverage

class TestCalcDistance(unittest.TestCase):
    def test_calc_distance_1(self):
        feature1 = np.array([1])
        feature2 = np.array([0])
        actual = calc_distance(feature1, feature2)
        self.assertEqual(actual, 1)

    def test_calc_distance_3(self):
        feature1 = np.array([3, 2, 1])
        feature2 = np.array([0, 0, 0])
        actual = calc_distance(feature1, feature2)
        self.assertEqual(actual, 14)

    def test_calc_distance_100(self):
        feature1 = np.ones(100)
        feature2 = np.zeros(100)
        actual = calc_distance(feature1, feature2)
        expected = feature1.sum()
        self.assertEqual(actual, expected)

