import math
import numpy as np
from sklearn import neighbors, svm

from sweeper import Board


# クローン認識器を表現するクラス
# このサンプルコードでは単純な 1-nearest neighbor 認識器とする（sklearnを使用）
# 下記と同型の fit メソッドと predict メソッドが必要
class LV1UserDefinedClassifierSVM:

    def __init__(self, n=-1):

        if n > 10:  # ラベル数が2未満だとsvmがエラーになるため
            self.clf = svm.SVC(C=10, gamma=10)
        else:
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


def lv1_user_function_sampling_sweeper(n_samples, target_model, exe_n):
    board_size = math.ceil(math.sqrt(exe_n))
    if n_samples < 0:
        raise ValueError

    elif n_samples == 0:
        return np.zeros((0, 2))

    elif n_samples == 1:
        new_board = Board(board_size=board_size)

        new_features = np.zeros((1, 2))

        feature_x, feature_y = new_board.get_optimal_solution()  # 最適解
        new_features[0][0] = feature_x
        new_features[0][1] = feature_y

        # target識別器からtargetのラベルを取得
        target_labels = target_model.predict(new_features)
        new_board.open_once_feature(feature_x=feature_x, feature_y=feature_y, color=target_labels[-1])  # 開示

        if n_samples == exe_n:
            return np.float32(new_features)
        else:
            return np.float32(new_features), new_board

    elif n_samples > 1:
        old_features, board = lv1_user_function_sampling_sweeper(n_samples=n_samples - 1, target_model=target_model,
                                                                 exe_n=exe_n)

        new_features = np.zeros((1, 2))

        feature_x, feature_y = board.get_optimal_solution()  # 最適解
        new_features[0][0] = feature_x
        new_features[0][1] = feature_y

        # target識別器からtargetのラベルを取得
        target_labels = target_model.predict(new_features)
        board.open_once_feature(feature_x=feature_x, feature_y=feature_y, color=target_labels[-1])

        features = np.vstack((old_features, new_features))

        if n_samples == exe_n:
            return np.float32(features)
        else:
            return np.float32(features), board


def lv1_user_function_sampling_sweeper_colorless(n_samples, target_model, exe_n):
    board_size = math.ceil(math.sqrt(exe_n))

    if n_samples < 0:
        raise ValueError

    elif n_samples == 0:
        return np.zeros((0, 2))

    elif n_samples == 1:

        new_board = Board(board_size=board_size)

        new_features = np.zeros((1, 2))

        feature_x, feature_y = new_board.get_optimal_solution()  # 最適解
        new_features[0][0] = feature_x
        new_features[0][1] = feature_y

        return np.float32(new_features)

    elif n_samples > 1:

        old_features = lv1_user_function_sampling_sweeper_colorless(n_samples=n_samples - 1, target_model=target_model,
                                                                    exe_n=exe_n)

        new_board = Board(board_size=board_size)

        # target識別器からtargetのラベルを取得
        target_labels = target_model.predict(old_features)

        # clone識別器からcloneのラベルを取得
        clone = LV1UserDefinedClassifierSVM(n=n_samples)
        # 学習
        clone.fit(features=old_features, labels=target_labels)
        clone_labels = clone.predict(features=old_features)

        for old_feature, target_label, clone_label in zip(old_features, target_labels, clone_labels):

            if target_label == clone_label:
                new_board.open_once_feature(feature_x=old_feature[0], feature_y=old_feature[1],
                                            color=target_label)  # 開示
            else:
                new_board.open_once_colorless_feature(feature_x=old_feature[0], feature_y=old_feature[1])  # 開示

        feature_x, feature_y = new_board.get_optimal_solution()  # 最適解

        new_features = np.zeros((1, 2))
        new_features[0][0] = feature_x
        new_features[0][1] = feature_y

        features = np.vstack((old_features, new_features))

        return np.float32(features)
