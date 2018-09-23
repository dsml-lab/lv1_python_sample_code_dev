import math

import numpy as np

from democ.distance import find_furthest_place, get_furthest_rate_arr
from democ.lv1_clf import LV1UserDefinedClassifierMLP1000HiddenLayer
from democ.lv2_clf import LV2UserDefinedClassifierMLP1000HiddenLayer
from democ.lv3_clf import LV3UserDefinedClassifier, LV3UserDefinedClassifierKNN3
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
    def create_lv3_voters(n_labels):
        voters = [Lv3Voter(model=LV3UserDefinedClassifier(n_labels=n_labels)),
                  Lv3Voter(model=LV3UserDefinedClassifierKNN3(n_labels=n_labels))]
        return voters

    def __init__(self, samplable_features, voter1: Voter, voter2: Voter):
        self.voter1 = voter1
        self.voter2 = voter2
        self.samplable_features = samplable_features

    def get_optimal_solution(self, sampled_features):
        self.predict_to_voters()

        discrepancy_rate_arr = self.get_discrepancy_rate_arr()
        furthest_rate_arr = get_furthest_rate_arr(sampled_features=sampled_features,
                                                  samplable_features=self.samplable_features)

        effective_distribution = discrepancy_rate_arr + furthest_rate_arr
        optimal_feature = np.amax(effective_distribution)

        self.delete_samplable_features(delete_feature=optimal_feature)

        return optimal_feature

    def delete_samplable_features(self, delete_feature):
        # # サンプリング候補から除外
        for i, able_fea in enumerate(self.samplable_features):
            if able_fea[0] == delete_feature[0]:
                del self.samplable_features[i]
                break

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
