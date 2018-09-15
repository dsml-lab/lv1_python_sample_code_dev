from sklearn.neural_network import MLPClassifier
import numpy as np


# クローン認識器を表現するクラス
# このサンプルコードでは各クラスラベルごとに単純な 5-nearest neighbor を行うものとする（sklearnを使用）
# 下記と同型の fit メソッドと predict_proba メソッドが必要
class LV2UserDefinedClassifierMLP1000HiddenLayer:

    def __init__(self, n_labels):
        self.n_labels = n_labels
        self.clfs = []
        for i in range(0, self.n_labels):
            clf = MLPClassifier(solver="lbfgs", hidden_layer_sizes=1000)
            self.clfs.append(clf)

    # クローン認識器の学習
    #   (features, likelihoods): 訓練データ（特徴量と尤度ベクトルのペアの集合）
    def fit(self, features, likelihoods):
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
        return np.float32(likelihoods)
