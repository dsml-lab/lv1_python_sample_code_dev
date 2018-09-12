import unittest

import numpy as np
import random


def calc_distance(feature1, feature2):
    feature_diff = feature1 - feature2
    feature_square = feature_diff**2
    return np.sum(feature_square)


def find_furthest_place(sampled_features, filtered_samplable_features):
    distance_arr = np.zeros((len(filtered_samplable_features), len(sampled_features)))

    for i, filtered_feature in enumerate(filtered_samplable_features):
        for j, sampled_feature in enumerate(sampled_features):
            distance_arr[i][j] = calc_distance(feature1=filtered_feature, feature2=sampled_feature)

    nearest_arr = np.zeros((len(filtered_samplable_features)))
    for i, filtered_feature in enumerate(filtered_samplable_features):
        nearest_arr[i] = np.min(distance_arr[i])

    print(nearest_arr)

    max_value = np.amax(nearest_arr)

    print(max_value)

    print("sampling候補数: " + str(len(nearest_arr)))

    index_list = np.where(max_value == nearest_arr)[0]
    random.shuffle(index_list)

    print(index_list)

    return filtered_samplable_features[index_list[0]]


def find_non_close_place(sampled_features, filtered_samplable_features):
    distance_arr = np.zeros((len(filtered_samplable_features), len(sampled_features)))

    for i, filtered_feature in enumerate(filtered_samplable_features):
        for j, sampled_feature in enumerate(sampled_features):
            distance_arr[i][j] = calc_distance(feature1=filtered_feature, feature2=sampled_feature)

    nearest_arr = np.zeros((len(filtered_samplable_features)))
    for i, filtered_feature in enumerate(filtered_samplable_features):
        nearest_arr[i] = np.min(distance_arr[i])

    print(nearest_arr)

    median_value = np.median(nearest_arr)

    print('中央値:' + str(median_value))

    print("sampling候補数: " + str(len(nearest_arr)))

    index_list = np.where(median_value == nearest_arr)[0]
    random.shuffle(index_list)

    print(index_list)

    return filtered_samplable_features[index_list[0]]


class DistanceTest(unittest.TestCase):

    def test_calc_distance(self):

        feature1 = np.array([2, 3])
        feature2 = np.array([4, 6])

        result = calc_distance(feature1=feature1, feature2=feature2)

        self.assertEqual(13, result)

    def test_find_furthest_place(self):
        feature1 = np.array([100, 99])
        feature2 = np.array([4, 6])
        feature3 = np.array([3, 6])

        feature4 = np.array([100, 100])
        feature5 = np.array([33, -55])

        sampled_features = [feature1, feature2, feature3]
        filtered_samplable_features = [feature4, feature5]

        result = find_furthest_place(sampled_features=sampled_features, filtered_samplable_features=filtered_samplable_features)

        self.assertEqual(0, result)

    def test_find_furthest_place_(self):
        sampled_features = [
            np.array([1, 1]),
            np.array([1, 1]),
            np.array([1, 1]),
            np.array([1, 1]),
            np.array([1, 1]),
            np.array([1, 1]),
        ]
        filtered_samplable_features = [
            np.array([2, 2]),
            np.array([4, 4]),
            np.array([6, 6]),
            np.array([8, 8]),
        ]
        result = find_furthest_place(sampled_features=sampled_features, filtered_samplable_features=filtered_samplable_features)

        self.assertEqual(0, result)