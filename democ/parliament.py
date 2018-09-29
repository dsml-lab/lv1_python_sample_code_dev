import math
import random

import numpy as np
from tqdm import trange

from democ.distance import find_furthest_place, get_furthest_rate_arr_lv2, get_furthest_rate_arr_lv3
from democ.lv1_clf import LV1UserDefinedClassifierMLP1000HiddenLayer
from democ.lv2_clf import LV2UserDefinedClassifierMLP1000HiddenLayer
from democ.lv3_clf import LV3UserDefinedClassifier, LV3UserDefinedClassifierKNN3, \
    LV3UserDefinedClassifierDivide, LV3UserDefinedClassifierEnsemble
from democ.voter import Lv1Voter, Lv2Voter, Voter, Lv3Voter


class Parliament:
    """議会クラス"""

    @staticmethod
    def get_image_size(exe_n):
        return math.ceil(math.sqrt(exe_n)) + 100

    @staticmethod
    def get_samplable_features_2_dimension(image_size):
        h = image_size // 2
        point_count = image_size * image_size
        samplable_features = np.zeros((point_count, 2))
        for i in range(0, point_count):
            x = i % image_size
            y = i // image_size
            samplable_features[i][0] = np.float32((x - h) / h)
            samplable_features[i][1] = np.float32(-(y - h) / h)
        return np.float32(samplable_features)

    @staticmethod
    def create_lv1_voters():
        voters = [Lv1Voter(model=LV1UserDefinedClassifierMLP1000HiddenLayer(), label_size=10),
                  Lv1Voter(model=LV1UserDefinedClassifierMLP1000HiddenLayer(), label_size=10)]
        return voters

    @staticmethod
    def create_lv2_voters():
        voters = [Lv2Voter(model=LV2UserDefinedClassifierMLP1000HiddenLayer(8), label_size=8),
                  Lv2Voter(model=LV2UserDefinedClassifierMLP1000HiddenLayer(8), label_size=8)]
        return voters

    @staticmethod
    def create_lv3_voters(labels_all):
        voters = [Lv3Voter(model=LV3UserDefinedClassifierEnsemble(labels_all=labels_all)),
                  Lv3Voter(model=LV3UserDefinedClassifierEnsemble(labels_all=labels_all))]
        return voters

    def __init__(self, samplable_features, voter1: Voter, voter2: Voter):
        self.voter1 = voter1
        self.voter2 = voter2
        self.samplable_features = samplable_features

    def get_optimal_solution_lv2(self, sampled_features):
        self.predict_to_voters()

        discrepancy_rate_arr = self.get_discrepancy_rate_arr()
        furthest_rate_arr = get_furthest_rate_arr_lv2(sampled_features=sampled_features,
                                                      samplable_features=self.samplable_features)
        effective_distribution = (discrepancy_rate_arr + furthest_rate_arr) / 2

        opt_index = effective_distribution.argmax()
        opt_feature = self.samplable_features[opt_index]
        self.delete_samplable_feature_lv2(delete_samplable_index=opt_index)

        return opt_feature

    def get_optimal_solution_lv3(self, sampled_features, number_of_return):
        self.predict_to_voters()

        discrepancy_rate_arr = self.get_discrepancy_rate_arr()
        # furthest_rate_arr = get_furthest_rate_arr_lv3(sampled_features=sampled_features,
        #                                               samplable_features=self.samplable_features)
        # effective_distribution = (discrepancy_rate_arr + furthest_rate_arr) / 2
        effective_distribution = discrepancy_rate_arr

        arg_sort_list = np.argsort(-effective_distribution)  # 降順

        index_list = np.where(number_of_return > arg_sort_list)[0]

        optimal_features = []
        if number_of_return >= len(index_list):
            for index in index_list:
                opt_feature = self.samplable_features[index]
                optimal_features.append(opt_feature)
        else:
            raise ValueError

        self.delete_samplable_features_lv3(delete_features=optimal_features)

        return optimal_features

    def delete_samplable_feature_lv2(self, delete_samplable_index):
        # サンプリング候補から除外
        self.samplable_features = np.delete(self.samplable_features, delete_samplable_index, axis=0)

    def delete_samplable_features_lv3(self, delete_features):
        temp_list = []
        # # サンプリング候補から除外
        for i, able_feature in enumerate(self.samplable_features):
            stay_flag = True
            for delete_feature in delete_features:
                if able_feature[0] == delete_feature[0]:
                    stay_flag = False

            if stay_flag:
                temp_list.append(self.samplable_features[i])

        self.samplable_features = temp_list

    def fit_to_voters(self, sampled_features, sampled_likelihoods):
        self.voter1.sampled_fit(sampled_features=sampled_features, sampled_likelihoods=sampled_likelihoods)
        self.voter2.sampled_fit(sampled_features=sampled_features, sampled_likelihoods=sampled_likelihoods)

    def predict_to_voters(self):
        self.voter1.samplable_predict(samplable_features=self.samplable_features)
        self.voter2.samplable_predict(samplable_features=self.samplable_features)

    def get_discrepancy_rate_arr(self):
        # # すべての投票者の投票結果を集計
        # 識別結果1と2の差分をとる
        samplable_likelihoods_diff = np.absolute(
            self.voter1.get_samplable_likelihoods() - self.voter2.get_samplable_likelihoods())

        # 同じ点の値を合計し、1次元行列に変換
        diff_sum_list = samplable_likelihoods_diff.sum(axis=1)

        return diff_sum_list / np.amax(diff_sum_list)
