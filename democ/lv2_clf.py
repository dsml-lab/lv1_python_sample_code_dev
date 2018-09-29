import numpy as np
from sklearn.grid_search import GridSearchCV
from sklearn.neural_network import MLPClassifier
# クローン認識器を表現するクラス
# このサンプルコードでは各クラスラベルごとに単純な 5-nearest neighbor を行うものとする（sklearnを使用）
# 下記と同型の fit メソッドと predict_proba メソッドが必要
from tqdm import trange


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
        for i in trange(0, self.n_labels):
            p = self.clfs[i].predict_proba(features)
            likelihoods = np.hstack([likelihoods, np.c_[p[:, 1]]])
        likelihoods = likelihoods[:, 1:]
        return np.float32(likelihoods)


class LV2UserDefinedClassifierMLP1000HiddenLayerGridSearch:
    def __init__(self, n_labels):
        parameters2 = {
            'hidden_layer_sizes': [900, 1000, 1100, 1200, 1300, 1400, 1500, 1700, 1800, 2000, 2200, 2500, 3000],

        }

        parameters = {'solver': ['lbfgs'], 'max_iter': [500, 1000, 1500], 'alpha': 10.0 ** -np.arange(1, 7),
                      'hidden_layer_sizes': [700, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1700, 1800, 2000, 2200, 2500, 3000], 'random_state': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]}

        self.n_labels = n_labels
        self.grid_search = []
        for i in range(0, self.n_labels):
            clf = MLPClassifier(solver="lbfgs", hidden_layer_sizes=1000)
            grid_search = GridSearchCV(clf,  # 分類器を渡す
                                       param_grid=parameters,  # 試行してほしいパラメータを渡す
                                       cv=10,  # 10-Fold CV で汎化性能を調べる
                                       )
            self.grid_search.append(grid_search)

    # クローン認識器の学習
    #   (features, likelihoods): 訓練データ（特徴量と尤度ベクトルのペアの集合）
    def fit(self, features, likelihoods):
        labels = np.int32(likelihoods >= 0.5)  # 尤度0.5以上のラベルのみがターゲット認識器の認識結果であると解釈する
        # Bool to Int
        for i in range(0, self.n_labels):
            l = labels[:, i]
            self.grid_search[i].fit(features, l)

            print('最大スコア')
            print(self.grid_search[i].best_score_)  # 最も良かったスコア
            print('最適パラメタ')
            print(self.grid_search[i].best_params_)  # 上記を記録したパラメータの組み合わせ

    # 未知の二次元特徴量を認識
    #   features: 認識対象の二次元特徴量の集合
    def predict_proba(self, features):
        likelihoods = np.c_[np.zeros(features.shape[0])]
        for i in trange(0, self.n_labels):
            p = self.clfs[i].predict_proba(features)
            likelihoods = np.hstack([likelihoods, np.c_[p[:, 1]]])
        likelihoods = likelihoods[:, 1:]
        return np.float32(likelihoods)
