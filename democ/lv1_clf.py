import numpy as np
from sklearn import svm, neighbors, tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.neural_network import MLPClassifier
from tqdm import tqdm


class LV1UserDefinedClassifierSVM10C10Gamma:
    def __init__(self):
        self.svm = svm.SVC(C=10, gamma=10)
        self.knn1 = neighbors.KNeighborsClassifier(n_neighbors=1)
        self.clf = None

    # クローン認識器の学習
    #   (features, labels): 訓練データ（特徴量とラベルのペアの集合）
    def fit(self, features, labels):

        if len(set(labels)) > 1:  # labelsが2以上ならSVM
            self.clf = self.svm
        else:
            self.clf = self.knn1

        self.clf.fit(features, labels)

    # 未知の二次元特徴量を認識
    #   features: 認識対象の二次元特徴量の集合
    def predict(self, features):
        labels = self.clf.predict(features)
        return np.int32(labels)


class LV1UserDefinedClassifierSVM10C10GammaGridSearch:
    def __init__(self):
        tuned_parameters = [
            {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
            {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.001, 0.0001]},
            {'C': [1, 10, 100, 1000], 'kernel': ['poly'], 'degree': [2, 3, 4], 'gamma': [0.001, 0.0001]},
            {'C': [1, 10, 100, 1000], 'kernel': ['sigmoid'], 'gamma': [0.001, 0.0001]}
        ]
        self.score = 'accuracy'
        self.svm = GridSearchCV(
            svm.SVC(),  # 識別器
            tuned_parameters,  # 最適化したいパラメータセット
            cv=5,  # 交差検定の回数
            scoring=self.score)  # モデルの評価関数の指定
        self.knn1 = neighbors.KNeighborsClassifier(n_neighbors=1)
        self.clf = None

    # クローン認識器の学習
    #   (features, labels): 訓練データ（特徴量とラベルのペアの集合）
    def fit(self, features, labels):

        if len(set(labels)) > 1:  # labelsが2以上ならSVM
            self.clf = self.svm
            self.clf.fit(features, labels)

            print("# Tuning hyper-parameters for %s" % self.score)
            print()
            print("Best parameters set found on development set: %s" % self.clf.best_params_)
            print()

            # それぞれのパラメータでの試行結果の表示
            print("Grid scores on development set:")
            print()
            for params, mean_score, scores in self.clf.grid_scores_:
                print("%0.3f (+/-%0.03f) for %r"
                      % (mean_score, scores.std() * 2, params))
            print()
        else:
            self.clf = self.knn1
            self.clf.fit(features, labels)

    # 未知の二次元特徴量を認識
    #   features: 認識対象の二次元特徴量の集合
    def predict(self, features):
        labels = self.clf.predict(features)

        return np.int32(labels)


class LV1UserDefinedClassifier1NN:
    def __init__(self):
        self.clf = neighbors.KNeighborsClassifier(n_neighbors=1)

    # クローン認識器の学習
    #   (features, labels): 訓練データ（特徴量とラベルのペアの集合）
    def fit(self, features, labels):
        self.clf.fit(features, labels)

    # 未知の二次元特徴量を認識
    #   features: 認識対象の二次元特徴量の集合
    def predict(self, features):
        labels = self.clf.predict(features)
        return np.int32(labels)


class LV1UserDefinedClassifier1NNRetry:
    def __init__(self):
        self.clf = neighbors.KNeighborsClassifier(n_neighbors=1)

    # クローン認識器の学習
    #   (features, labels): 訓練データ（特徴量とラベルのペアの集合）
    def fit(self, features, labels):
        while True:
            self.clf.fit(features, labels)
            predict_labels = self.clf.predict(features)
            if np.allclose(labels, predict_labels):
                print('一致')
                break
            else:
                print('不一致')

    # 未知の二次元特徴量を認識
    #   features: 認識対象の二次元特徴量の集合
    def predict(self, features):
        labels = self.clf.predict(features)
        return np.int32(labels)


class LV1UserDefinedClassifier7NN:
    def __init__(self):
        self.knn1 = neighbors.KNeighborsClassifier(n_neighbors=1)
        self.knn7 = neighbors.KNeighborsClassifier(n_neighbors=7)
        self.clf = None

    # クローン認識器の学習
    #   (features, labels): 訓練データ（特徴量とラベルのペアの集合）
    def fit(self, features, labels):
        if len(features) > 7:
            self.clf = self.knn7
        else:
            self.clf = self.knn1

        self.clf.fit(features, labels)

    # 未知の二次元特徴量を認識
    #   features: 認識対象の二次元特徴量の集合
    def predict(self, features):
        labels = self.clf.predict(features)
        return np.int32(labels)


class LV1UserDefinedClassifierMLP1000HiddenLayer:
    def __init__(self):
        self.clf = MLPClassifier(solver="lbfgs", hidden_layer_sizes=1000)

    # クローン認識器の学習
    #   (features, labels): 訓練データ（特徴量とラベルのペアの集合）
    def fit(self, features, labels):
        self.clf.fit(features, labels)

    # 未知の二次元特徴量を認識
    #   features: 認識対象の二次元特徴量の集合
    def predict(self, features):
        labels = self.clf.predict(features)
        return np.int32(labels)


class LV1UserDefinedClassifierMLP1000HiddenLayerCorrectLabels:
    def __init__(self):
        self.clf = MLPClassifier(solver="lbfgs", hidden_layer_sizes=1000)
        self.sampled_features = None
        self.sampled_labels = None

    # クローン認識器の学習
    #   (features, labels): 訓練データ（特徴量とラベルのペアの集合）
    def fit(self, features, labels):
        self.sampled_features = features
        self.sampled_labels = labels
        self.clf.fit(features, labels)

    # 未知の二次元特徴量を認識
    #   features: 認識対象の二次元特徴量の集合
    def predict(self, features):
        labels = self.clf.predict(features)
        return np.int32(self.correct_labels(features=features, labels=labels))

    @staticmethod
    def convert_to_px_arr(arr):
        image_size = 512
        return np.int32((arr + 1.0) * 0.5 * image_size)

    def correct_labels(self, features, labels):
        # for i, feature in enumerate(tqdm(features)):
        #     for j, sampled in enumerate(self.sampled_features):
        #         if np.allclose(feature, sampled):
        #             labels[i] = self.sampled_labels[i]

        features_px = self.convert_to_px_arr(features)
        sampled_features_px = self.convert_to_px_arr(self.sampled_features)

        match_count = 0
        for i, sampled in enumerate(tqdm(sampled_features_px)):
            index_list = np.where(sampled == features_px)[0]
            # print(sampled)
            if len(index_list) != 0:
                print('一致')
                match_count += 1
                labels[index_list[0]] = self.sampled_labels[i]

        print('一致数: ' + str(match_count))

        return labels


class LV1UserDefinedClassifierTree1000MaxDepth:
    def __init__(self):
        self.clf = tree.DecisionTreeClassifier(max_depth=1000)

    # クローン認識器の学習
    #   (features, labels): 訓練データ（特徴量とラベルのペアの集合）
    def fit(self, features, labels):
        self.clf.fit(features, labels)

    # 未知の二次元特徴量を認識
    #   features: 認識対象の二次元特徴量の集合
    def predict(self, features):
        labels = self.clf.predict(features)
        return np.int32(labels)


class LV1UserDefinedClassifierRandomForest:
    def __init__(self):
        self.clf = RandomForestClassifier()

    # クローン認識器の学習
    #   (features, labels): 訓練データ（特徴量とラベルのペアの集合）
    def fit(self, features, labels):
        self.clf.fit(features, labels)

    # 未知の二次元特徴量を認識
    #   features: 認識対象の二次元特徴量の集合
    def predict(self, features):
        labels = self.clf.predict(features)
        return np.int32(labels)