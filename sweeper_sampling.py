import math
import sys

import numpy as np
from sklearn import neighbors, svm

from sweeper import Board
sys.setrecursionlimit(100000) # 再帰上限


# クローン認識器を表現するクラス
# このサンプルコードでは単純な 1-nearest neighbor 認識器とする（sklearnを使用）
# 下記と同型の fit メソッドと predict メソッドが必要
class LV1UserDefinedClassifierSVM:

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


# ターゲット認識器に入力する二次元特徴量をサンプリングする関数(格子・長方形)
#   n_samples: サンプリングする特徴量の数
def lv1_user_function_sampling_meshgrid_rectangular(n_samples):
    features = np.zeros((n_samples, 2))

    x_samples = 0
    y_samples = 0

    # 格子点の個数がもっとも多くなる
    # y_sizeとy_sizeの差がなるべく小さくなる

    for i in range(2, n_samples):
        for j in range(2, n_samples):
            if n_samples >= i * j > x_samples * y_samples and abs(i - j) < 2:  # 格子の縦横の差が2より小さい
                x_samples = i
                y_samples = j

    print('x_samples:' + str(x_samples))
    print('y_samples:' + str(y_samples))

    # 格子ひとつ分の幅
    x_size = 2 / (x_samples + 1)
    y_size = 2 / (y_samples + 1)

    # 格子状に値を入れる
    count = 0
    for j in range(1, x_samples + 1):
        for k in range(1, y_samples + 1):
            features[count][0] = j * x_size - 1
            features[count][1] = k * y_size - 1
            count = count + 1

    # 残りはランダムに
    for i in range(x_samples * y_samples, n_samples):
        features[i][0] = 2 * np.random.rand() - 1
        features[i][1] = 2 * np.random.rand() - 1
    return np.float32(features)


def lv1_user_function_sampling_sweeper(n_samples, target_model, exe_n):
    board_size_x = math.ceil(math.sqrt(exe_n))
    board_size_y = math.ceil(math.sqrt(exe_n))

    if n_samples < 0:
        raise ValueError

    elif n_samples == 0:
        return np.zeros((0, 2))

    elif n_samples == 1:
        new_board = Board(board_size_x=board_size_x, board_size_y=board_size_y)

        print('n_samples:' + str(n_samples) + ', ' + 'exe_n:' + str(exe_n))

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
        old_features, old_board = lv1_user_function_sampling_sweeper(n_samples=n_samples - 1, target_model=target_model,
                                                                 exe_n=exe_n)

        print('n_samples:' + str(n_samples) + ', ' + 'exe_n:' + str(exe_n))

        new_features = np.zeros((1, 2))

        feature_x, feature_y = old_board.get_optimal_solution()  # 最適解
        new_features[0][0] = feature_x
        new_features[0][1] = feature_y

        # target識別器からtargetのラベルを取得
        target_labels = target_model.predict(new_features)
        old_board.open_once_feature(feature_x=feature_x, feature_y=feature_y, color=target_labels[-1])

        features = np.vstack((old_features, new_features))

        if n_samples == exe_n:
            return np.float32(features)
        else:
            return np.float32(features), old_board


def lv1_user_function_sampling_sweeper_pixel(n_samples, target_model, exe_n, before_features):
    board_size_x = 256
    board_size_y = 256

    if n_samples < 0:
        raise ValueError

    elif n_samples == 0:
        return np.zeros((0, 2))

    elif n_samples == 1:
        new_board = Board(board_size_x=board_size_x, board_size_y=board_size_y)

        before_labels = target_model.predict(before_features)
        for before_feature, before_label in zip(before_features, before_labels):
            new_board.open_once_feature(feature_x=before_feature[0], feature_y=before_feature[1], color=before_label)

        print('n_samples:' + str(n_samples) + ', ' + 'exe_n:' + str(exe_n))

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
        old_features, old_board = lv1_user_function_sampling_sweeper_pixel(n_samples=n_samples - 1, target_model=target_model,
                                                                 exe_n=exe_n)

        print('n_samples:' + str(n_samples) + ', ' + 'exe_n:' + str(exe_n))

        new_features = np.zeros((1, 2))

        feature_x, feature_y = old_board.get_optimal_solution()  # 最適解
        new_features[0][0] = feature_x
        new_features[0][1] = feature_y

        # target識別器からtargetのラベルを取得
        target_labels = target_model.predict(new_features)
        old_board.open_once_feature(feature_x=feature_x, feature_y=feature_y, color=target_labels[-1])

        features = np.vstack((old_features, new_features))

        if n_samples == exe_n:
            return np.float32(features)
        else:
            return np.float32(features), old_board



def lv1_user_function_sampling_sweeper_colorless(n_samples, target_model, exe_n):
    board_size_x = math.ceil(math.sqrt(exe_n))
    board_size_y = math.ceil(math.sqrt(exe_n))

    if n_samples < 0:
        raise ValueError

    elif n_samples == 0:
        return np.zeros((0, 2))

    elif n_samples == 1:

        new_board = Board(board_size_x=board_size_x, board_size_y=board_size_y)

        print('n_samples:' + str(n_samples) + ', ' + 'exe_n:' + str(exe_n))

        new_features = np.zeros((1, 2))

        feature_x, feature_y = new_board.get_optimal_solution()  # 最適解
        new_features[0][0] = feature_x
        new_features[0][1] = feature_y

        return np.float32(new_features)

    elif n_samples > 1:

        old_features = lv1_user_function_sampling_sweeper_colorless(n_samples=n_samples - 1, target_model=target_model,
                                                                    exe_n=exe_n)

        print('n_samples:' + str(n_samples) + ', ' + 'exe_n:' + str(exe_n))

        new_board = Board(board_size_x=board_size_x, board_size_y=board_size_y)

        # target識別器からtargetのラベルを取得
        target_labels = target_model.predict(old_features)

        # clone識別器からcloneのラベルを取得
        clone = LV1UserDefinedClassifierSVM()
        # 学習
        clone.fit(features=old_features, labels=target_labels)
        clone_labels = clone.predict(features=old_features)

        for old_feature, target_label, clone_label in zip(old_features, target_labels, clone_labels):

            if target_label == clone_label:
                new_board.open_once_feature(feature_x=old_feature[0], feature_y=old_feature[1],
                                            color=target_label)  # 開示
            else:
                new_board.open_once_colorless_feature(feature_x=old_feature[0], feature_y=old_feature[1],
                                                      color=target_label)  # 開示

        feature_x, feature_y = new_board.get_optimal_solution()  # 最適解

        new_features = np.zeros((1, 2))
        new_features[0][0] = feature_x
        new_features[0][1] = feature_y

        features = np.vstack((old_features, new_features))

        return np.float32(features)


def lv1_user_function_sampling_sweeper_or_grid_or_grid_edge(n_samples, target_model):
    small_threshold = 32
    large_threshold = 256

    if small_threshold > n_samples:
        return lv1_user_function_sampling_sweeper_colorless(n_samples=n_samples, exe_n=n_samples, target_model=target_model)
    elif large_threshold > n_samples:
        return lv1_user_function_sampling_meshgrid_rectangular(n_samples=n_samples)
    else:
        edge_n_samples = n_samples - large_threshold
        grid_n_samples = large_threshold

        grid_features = lv1_user_function_sampling_meshgrid_rectangular(n_samples=grid_n_samples)
        edge_features = lv1_user_function_sampling_sweeper_pixel(n_samples=edge_n_samples, exe_n=edge_n_samples, target_model=target_model, before_features=grid_features)

        return np.vstack((edge_features, grid_features))

        # return lv1_user_function_sampling_sweeper_colorless(n_samples=n_samples, exe_n=n_samples, target_model=target_model)

