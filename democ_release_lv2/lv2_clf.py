import unittest

from sklearn.neural_network import MLPClassifier
import numpy as np


# クローン認識器を表現するクラス
class LV2UserDefinedClassifierMLP1000HiddenLayerCorrectLabels:

    def __init__(self, n_labels):
        self.n_labels = n_labels
        self.clfs = []
        for i in range(0, self.n_labels):
            clf = MLPClassifier(solver="lbfgs", hidden_layer_sizes=1000, activation='relu', learning_rate="invscaling")
            self.clfs.append(clf)
        self.sampled_features = None
        self.sampled_likelihoods = None

    # クローン認識器の学習
    #   (features, likelihoods): 訓練データ（特徴量と尤度ベクトルのペアの集合）
    def fit(self, features, likelihoods):
        self.sampled_features = features
        self.sampled_likelihoods = likelihoods
        labels = np.int32(likelihoods >= 0.5)  # 尤度0.5以上のラベルのみがターゲット認識器の認識結果であると解釈する
        # Bool to Int
        for i in range(0, self.n_labels):
            l = labels[:, i]
            self.clfs[i].fit(features, l)

    # 未知の二次元特徴量を認識
    #   features: 認識対象の二次元特徴量の集合
    def predict_proba(self, features):
        likelihoods = np.c_[np.zeros(features.shape[0])]
        for i in range(0, self.n_labels):
            p = self.clfs[i].predict_proba(features)
            likelihoods = np.hstack([likelihoods, np.c_[p[:, 1]]])
        likelihoods = likelihoods[:, 1:]
        return likelihoods
        # return np.float32(self.correct_labels(features, likelihoods))

    @staticmethod
    def convert_to_px_arr(arr):
        image_size = 512
        return np.int32((arr + 1.0) * 0.5 * image_size)

    @staticmethod
    def convert_unique_value(px_arr):
        px_arr[:, 0] = px_arr[:, 0] * 1000

    # 正しいlikelihoodsを返す
    def correct_labels(self, features, likelihoods):
        features_px = self.convert_to_px_arr(features)
        sampled_features_px = self.convert_to_px_arr(self.sampled_features)

        features_px[:, 0] = features_px[:, 0] * 1000
        sampled_features_px[:, 0] = sampled_features_px[:, 0] * 1000

        features_px = np.sum(features_px, axis=1)
        sampled_features_px = np.sum(sampled_features_px, axis=1)

        for i, sampled in enumerate(sampled_features_px):
            index_list = np.where(sampled == features_px)[0]
            if len(index_list) > 0:
                likelihoods[index_list[0]] = self.sampled_likelihoods[i]

        return likelihoods


# クローン認識器を表現するクラス
class LV2UserDefinedClassifierMLP1700HiddenLayerCorrectLabels:

    def __init__(self, n_labels):
        self.n_labels = n_labels
        self.clfs = []
        for i in range(0, self.n_labels):
            clf = MLPClassifier(solver="lbfgs", hidden_layer_sizes=1700, activation='relu', learning_rate="constant")
            self.clfs.append(clf)
        self.sampled_features = None
        self.sampled_likelihoods = None

    # クローン認識器の学習
    #   (features, likelihoods): 訓練データ（特徴量と尤度ベクトルのペアの集合）
    def fit(self, features, likelihoods):
        self.sampled_features = features
        self.sampled_likelihoods = likelihoods
        labels = np.int32(likelihoods >= 0.5)  # 尤度0.5以上のラベルのみがターゲット認識器の認識結果であると解釈する
        # Bool to Int
        for i in range(0, self.n_labels):
            l = labels[:, i]
            self.clfs[i].fit(features, l)

    # 未知の二次元特徴量を認識
    #   features: 認識対象の二次元特徴量の集合
    def predict_proba(self, features):
        likelihoods = np.c_[np.zeros(features.shape[0])]
        for i in range(0, self.n_labels):
            p = self.clfs[i].predict_proba(features)
            likelihoods = np.hstack([likelihoods, np.c_[p[:, 1]]])
        likelihoods = likelihoods[:, 1:]
        return likelihoods
        # return np.float32(self.correct_labels(features, likelihoods))

    @staticmethod
    def convert_to_px_arr(arr):
        image_size = 512
        return np.int32((arr + 1.0) * 0.5 * image_size)

    @staticmethod
    def convert_unique_value(px_arr):
        px_arr[:, 0] = px_arr[:, 0] * 1000

    # 正しいlikelihoodsを返す
    def correct_labels(self, features, likelihoods):
        features_px = self.convert_to_px_arr(features)
        sampled_features_px = self.convert_to_px_arr(self.sampled_features)

        features_px[:, 0] = features_px[:, 0] * 1000
        sampled_features_px[:, 0] = sampled_features_px[:, 0] * 1000

        features_px = np.sum(features_px, axis=1)
        sampled_features_px = np.sum(sampled_features_px, axis=1)

        for i, sampled in enumerate(sampled_features_px):
            index_list = np.where(sampled == features_px)[0]
            if len(index_list) > 0:
                likelihoods[index_list[0]] = self.sampled_likelihoods[i]

        return likelihoods


class LV2UserDefinedClassifierMLP1000HiddenLayerCorrectLabelsTest(unittest.TestCase):

    def test_correct_labels(self):

        n_labels = 8
        n_samples = 2

        model = LV2UserDefinedClassifierMLP1000HiddenLayerCorrectLabels(n_labels=n_labels)

        sampled_features = np.zeros((n_samples, 2))
        sampled_features[0][0] = 0.1
        sampled_features[0][1] = 0.2
        sampled_features[1][0] = 0.3
        sampled_features[1][1] = 0.3

        features = np.zeros((10, 2))
        # 一致
        features[5][0] = 0.1
        features[5][1] = 0.2
        # 不一致
        features[7][0] = 0.3
        features[7][1] = 0.3

        likelihoods = np.zeros((10, n_labels))
        original_likelihoods = np.zeros((10, n_labels))

        sampled_likelihoods = np.ones((n_samples, n_labels))
        sampled_likelihoods[0] = np.full_like(n_labels, 5)
        sampled_likelihoods[1] = np.full_like(n_labels, 6)

        model.sampled_features = sampled_features
        model.sampled_likelihoods = sampled_likelihoods
        result = model.correct_labels(features=features, likelihoods=likelihoods)

        print('result')
        print(result)

        print(sampled_likelihoods[0][0])
        print(likelihoods)

        print(result[0][0])
        print(result[1][0])

        self.assertEqual(result[0][0], original_likelihoods[0][0])
        self.assertEqual(result[7][0], sampled_likelihoods[1][0])
