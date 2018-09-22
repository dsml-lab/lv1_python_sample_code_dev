import math

import numpy as np

from democ.distance import find_furthest_place
from democ.lv1_clf import LV1UserDefinedClassifierMLP1000HiddenLayer
from democ.lv2_clf import LV2UserDefinedClassifierMLP1000HiddenLayer
from democ.lv3_clf import LV3UserDefinedClassifier, LV3UserDefinedClassifierKNN7
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
                  Lv3Voter(model=LV3UserDefinedClassifierKNN7(n_labels=n_labels))]
        return voters

    def __init__(self, samplable_features, voter1: Voter, voter2: Voter):
        self.voter1 = voter1
        self.voter2 = voter2
        self.samplable_features = samplable_features

    def get_optimal_solution(self, sampled_features, n_labels):
        self.predict_to_voters()

        filtered_samplable_features = self.calc_filtered_samplable_features(n_labels)

        opt_index = find_furthest_place(sampled_features=sampled_features,
                                        filtered_samplable_features=filtered_samplable_features)

        self.delete_samplable_features(delete_feature=filtered_samplable_features[opt_index])

        return filtered_samplable_features[opt_index]

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

    def calc_filtered_samplable_features(self, n_labels):
        # # すべての投票者の投票結果を集計
        # 識別結果1と2の差分をとる
        samplable_likelihoods_diff = np.absolute(
            self.voter1.get_samplable_likelihoods() - self.voter2.get_samplable_likelihoods())

        # 同じ点の値を合計し、1次元行列に変換
        wrong_ratio_list = samplable_likelihoods_diff.sum(axis=1) / n_labels
        predict_result_is_match_list = np.int32(wrong_ratio_list >= 0.8)  # ラベルが80%以上異なっている点をサンプリング対象とする

        max_value = np.amax(predict_result_is_match_list)
        index_list = np.where(predict_result_is_match_list == max_value)[0]  # 識別見解が一致しない点を抽出

        print('samplable_likelihoods_diff:')
        print(np.unique(samplable_likelihoods_diff))

        print('wrong_ratio_list:')
        print(np.unique(wrong_ratio_list))

        print('predict_result_is_match_list:')
        print(np.unique(predict_result_is_match_list))

        # filtered_samplable_features = self.samplable_features[index_list]
        filtered_samplable_features = []
        for index in index_list:
            filtered_samplable_features.append(self.samplable_features[index])

        return filtered_samplable_features
