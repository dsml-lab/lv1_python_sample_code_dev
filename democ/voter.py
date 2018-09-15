import numpy as np
from sklearn.preprocessing import OneHotEncoder


class Lv1Voter:
    """投票者クラス"""

    def __init__(self, model, label_size):
        self.model = model
        self.samplable_labels = None
        self.label_size = label_size

    # クローン認識器の学習
    #   (features, labels): 訓練データ（特徴量とラベルのペアの集合）
    def sampled_fit(self, sampled_features, sampled_labels):
        self.model.fit(sampled_features, sampled_labels)  # 学習

    # 未知の二次元特徴量を認識
    #   features: 認識対象の二次元特徴量の集合
    def samplable_predict(self, samplable_features):
        self.samplable_labels = self.__to_one_hot(self.model.predict(samplable_features))  # 予測結果を保持

    def __to_one_hot(self, labels):
        encoder = OneHotEncoder(self.label_size)
        return encoder.fit_transform(np.reshape(labels, (-1, 1))).toarray()


class Lv2Voter:
    """投票者クラス"""

    def __init__(self, model, label_size):
        self.model = model
        self.samplable_labels = None
        self.label_size = label_size

    # クローン認識器の学習
    #   (features, labels): 訓練データ（特徴量とラベルのペアの集合）
    def sampled_fit(self, sampled_features, sampled_labels):
        self.model.fit(features=sampled_features, likelihoods=sampled_labels)  # 学習

    # 未知の二次元特徴量を認識
    #   features: 認識対象の二次元特徴量の集合
    def samplable_predict(self, samplable_features):
        likelihoods = self.model.predict_proba(samplable_features)
        self.samplable_labels = np.int32(likelihoods >= 0.5)  # 予測結果を保持
