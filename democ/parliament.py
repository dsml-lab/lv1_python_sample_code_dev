import math
import numpy as np

from democ.distance import find_furthest_place
from democ.lv1_clf import LV1UserDefinedClassifierMLP1000HiddenLayer
from democ.lv2_clf import LV2UserDefinedClassifierMLP1000HiddenLayer
from democ.voter import Lv1Voter, Lv2Voter


class Parliament:
    """議会クラス"""

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

    def __init__(self, dimension, label_size, samplable_features, voters):
        self.voters = voters
        self.dimension = dimension
        self.label_size = label_size
        self.samplable_features = samplable_features

    def get_optimal_solution(self, sampled_features, sampled_labels):
        self.__fit_to_voters(sampled_features=sampled_features, sampled_labels=sampled_labels)  # 投票者を訓練
        self.__predict_to_voters()  # 投票者による予測

        # サンプリング特徴量候補数,ラベル(one hot形式)数の集計用行列を作成
        # label_count_arr = np.zeros((len(self.samplable_features), self.label_size))

        # すべての投票者の投票結果を集計
        # for voter in self.voters:
        #     samplable_labels = voter.samplable_labels  # 投票者のサンプリング特徴量候補の予測結果(labels)を取得
        #
        #     label_count_arr = label_count_arr + samplable_labels  # 投票者のぶんだけ、予測結果[0,0,1,0...0]を集計用行列に加算
        #
        # label_count_arr[label_count_arr == len(self.voters)] = 0  # すべての識別器が同じ予測結果
        #
        # label_count_arr = label_count_arr.sum(axis=1)  # 同じ点の値を合計し、1次元行列に変換
        # label_count_arr[label_count_arr > 0] = 1  # 1以上の値は1に変更

        # 識別結果1と2の差分をとる
        label_count_arr = np.absolute(self.voters[0].samplable_labels - self.voters[1].samplable_labels)
        # 同じ点の値を合計し、1次元行列に変換
        label_count_arr = label_count_arr.max(axis=1)
        # label_count_arr = label_count_arr.sum(axis=1)

        max_value = np.amax(label_count_arr)
        filtered_index_list = np.where(label_count_arr == max_value)[0]
        filtered_samplable_features = self.samplable_features[filtered_index_list]
        # out_of_range_samplable_features = self.samplable_features[filtered_index_list]

        opt_feature = find_furthest_place(sampled_features=sampled_features,
                                          filtered_samplable_features=filtered_samplable_features)

        self.delete_samplable_features(delete_feature=opt_feature)

        return opt_feature

    def delete_samplable_features(self, delete_feature):
        index_list = np.where(delete_feature == self.samplable_features)[0]

        # サンプリング候補から除外
        self.samplable_features = np.delete(self.samplable_features, index_list[0], axis=0)

    def __fit_to_voters(self, sampled_features, sampled_labels):
        for i in range(len(self.voters)):
            self.voters[i].sampled_fit(sampled_features=sampled_features, sampled_labels=sampled_labels)

    def __predict_to_voters(self):
        for i in range(len(self.voters)):
            self.voters[i].samplable_predict(samplable_features=self.samplable_features)