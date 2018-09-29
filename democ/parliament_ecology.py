import math
import random

import numpy as np
from tqdm import trange

from democ.distance import find_furthest_place, get_furthest_rate_arr_lv2, get_furthest_rate_arr_lv3
from democ.lv1_clf import LV1UserDefinedClassifierMLP1000HiddenLayer
from democ.lv2_clf import LV2UserDefinedClassifierMLP1000HiddenLayer
from democ.lv3_clf import LV3UserDefinedClassifier, LV3UserDefinedClassifierKNN3,\
    LV3UserDefinedClassifierDivide
from democ.voter import Lv1Voter, Lv2Voter, Voter, Lv3Voter


class ParliamentEcology:
    """議会クラス"""

    @staticmethod
    def create_lv3_voters(labels_all):
        voters = [Lv3Voter(model=LV3UserDefinedClassifierDivide(labels_all=labels_all)),
                  Lv3Voter(model=LV3UserDefinedClassifierDivide(labels_all=labels_all))]
        return voters

    def __init__(self, samplable_features, latest_voter, old_voter):
        self.samplable_features = samplable_features
        self.latest_voter = latest_voter
        self.old_voter = old_voter

    def get_optimal_solution_single(self, sampled_features):
        self.predict_to_voters()

        discrepancy_rate_arr = self.get_discrepancy_rate_arr()
        furthest_rate_arr = get_furthest_rate_arr_lv2(sampled_features=sampled_features,
                                                      samplable_features=self.samplable_features)
        effective_distribution = (discrepancy_rate_arr + furthest_rate_arr) / 2
        filter_effective_distribution_index = np.where(effective_distribution > 0.9)[0]
        random.shuffle(filter_effective_distribution_index)

        opt_index = effective_distribution.argmax()
        opt_feature = self.samplable_features[opt_index]
        self.delete_samplable_features_lv3(delete_features=[opt_feature])

        return opt_feature

    def get_optimal_solution_multi(self, number_of_return):
        self.predict_to_voters()

        discrepancy_rate_arr = self.get_discrepancy_rate_arr()
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
        self.latest_voter.sampled_fit(sampled_features=sampled_features, sampled_likelihoods=sampled_likelihoods)
        self.swap_voters()

    def swap_voters(self):
        temp = self.latest_voter
        self.latest_voter = self.old_voter
        self.old_voter = temp

    def predict_to_voters(self):
        self.latest_voter.samplable_predict(samplable_features=self.samplable_features)
        self.old_voter.samplable_predict(samplable_features=self.samplable_features)

    def get_discrepancy_rate_arr(self):
        # # すべての投票者の投票結果を集計
        # 識別結果1と2の差分をとる
        samplable_likelihoods_diff = np.absolute(
            self.latest_voter.get_samplable_likelihoods() - self.old_voter.get_samplable_likelihoods())

        # 同じ点の値を合計し、1次元行列に変換
        diff_sum_list = samplable_likelihoods_diff.sum(axis=1)

        return diff_sum_list / np.amax(diff_sum_list)
