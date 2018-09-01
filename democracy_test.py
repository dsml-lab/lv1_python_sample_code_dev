import unittest
import numpy as np

from democracy import to_one_hot, Voter, LV1UserDefinedClassifierSVM


class DemocracyTest(unittest.TestCase):

    def test_to_one_hot(self):

        labels = [[1], [2], [3], [9], [7]]

        result = to_one_hot(labels)

        expected_result = np.array([[1., 0., 0., 0., 0.],
                             [0., 1., 0., 0., 0.],
                             [0., 0., 1., 0., 0.],
                             [0., 0., 0., 0., 1.],
                             [0., 0., 0., 1., 0.]])
        print(expected_result)
        print(result)

        self.assertEqual(result.tolist(), expected_result.tolist())


class VoterTest(unittest.TestCase):

    def test_samplable_predict(self):

        features = np.zeros((3, 2))
        features[0][0] = 0.1
        features[0][1] = 0.2
        features[1][0] = 0.3
        features[1][1] = 0.4
        features[2][0] = 0.5
        features[2][1] = 0.6

        labels = np.zeros((3))
        labels[0] = 1
        labels[1] = 2
        labels[2] = 5

        samplable_features = np.zeros((5, 2))
        samplable_features[0][0] = 0.9
        samplable_features[0][1] = 0.8
        samplable_features[1][0] = 0.7
        samplable_features[1][1] = 0.6
        samplable_features[2][0] = 0.5
        samplable_features[2][1] = 0.4
        samplable_features[3][0] = 0.3
        samplable_features[3][1] = 0.2
        samplable_features[4][0] = 0.1
        samplable_features[4][1] = 0.0

        v = Voter(model=LV1UserDefinedClassifierSVM())
        v.sampled_fit(sampled_features=features, sampled_labels=np.int32(labels))
        v.samplable_predict(samplable_features=samplable_features)

        print(v.samplable_labels)