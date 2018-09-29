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

    max_value = np.amax(nearest_arr)

    print("sampling候補数: " + str(len(nearest_arr)))

    index_list = np.where(max_value == nearest_arr)[0]
    random.shuffle(index_list)

    return filtered_samplable_features[index_list[0]]
