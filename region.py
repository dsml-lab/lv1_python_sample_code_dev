from datetime import datetime

import os

import numpy as np
from sympy import Circle
from sympy.geometry import Polygon, Point
import matplotlib.pyplot as plt
from sympy.plotting import plot
from PIL import Image
from sklearn import neighbors
import math

import sys

from sympy.plotting.plot import Plot
from tqdm import tqdm

from labels import COLOR2ID, ID2COLOR
from sweeper import Board, COLORLESS

IMAGE_SIZE = 512
CLASS_SIZE = 10
DIVIDER = '------------------------'

sys.setrecursionlimit(10000)


class LV1TargetClassifier:

    def __init__(self):
        self.img = None

    # ターゲット認識器をロード
    #   filename: ターゲット認識器を表現する画像のファイルパス
    def load(self, filename):
        self.img = Image.open(filename)

    # 入力された二次元特徴量に対し，その認識結果（クラスラベルID）を返す
    def predict_once(self, x1, x2):
        h = IMAGE_SIZE // 2
        x = max(0, min(IMAGE_SIZE - 1, np.round(h * x1 + h)))
        y = max(0, min(IMAGE_SIZE - 1, np.round(h - h * x2)))
        return COLOR2ID(self.img.getpixel((x, y)))

    # 入力された二次元特徴量の集合に対し，各々の認識結果を返す
    def predict(self, features):
        labels = np.zeros(features.shape[0])
        for i in range(0, features.shape[0]):
            labels[i] = self.predict_once(features[i][0], features[i][1])
        return np.int32(labels)


# クローン認識器を表現するクラス
# このサンプルコードでは単純な 1-nearest neighbor 認識器とする（sklearnを使用）
# 下記と同型の fit メソッドと predict メソッドが必要
class LV1UserDefinedClassifier:

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


def create_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


class SavePathManager:

    def __init__(self, save_root_dir):
        self.save_root_dir = save_root_dir

    def exe_n_dir(self, exe_n):
        return os.path.join(self.save_root_dir, 'n_' + str(exe_n))

    def sampling_method_dir(self, exe_n, method_name):
        return os.path.join(self.exe_n_dir(exe_n), method_name)

    def sampling_history_dir(self, exe_n, method_name):
        return os.path.join(self.sampling_method_dir(exe_n=exe_n, method_name=method_name), 'history')

    def sampling_history_n_dir(self, exe_n, method_name, sampling_n):
        return os.path.join(self.sampling_history_dir(exe_n=exe_n, method_name=method_name), 'n_' + str(sampling_n))


def create_region_map(features, target_labels, n, clone_labels):
    print(DIVIDER)
    print('features: ')
    print(features)
    print(DIVIDER)
    print('target_labels: ')
    print(target_labels)
    print(DIVIDER)
    print('clone_labels: ')
    print(clone_labels)
    print(DIVIDER)

    # 各色がサンプリング点の中でいくつあったかを記録する配列
    color_counts = np.zeros(CLASS_SIZE, dtype=np.int)

    # 全ての点がそれぞれどの色かカウント
    for i in range(n):
        target_label = target_labels[i]
        clone_label = clone_labels[i]

        # ターゲットの識別結果とクローンの識別結果が同じ点のみ
        if target_label == clone_label:
            # 各色がサンプリング点の中でいくつあったか
            color_counts[target_label] = color_counts[target_label] + 1

    # 色の勢力ごとのfeaturesを作成
    color_features_list = []
    for i in range(CLASS_SIZE):
        color_features_list.append(np.zeros((color_counts[i], 2)))

    # 各々の色の配列に値を代入
    for num_color in range(CLASS_SIZE):
        count = 0
        for i in range(n):
            x = features[i][0]
            y = features[i][1]
            target_label = target_labels[i]
            clone_label = clone_labels[i]

            # ターゲットの識別結果とクローンの識別結果が同じ点かつ、対応する色の点のみ
            if target_label == clone_label == num_color:
                color_features_list[num_color][count][0] = x
                color_features_list[num_color][count][1] = y
                count = count + 1

    # for i, fea in enumerate(color_features_list):
    #     print(DIVIDER)
    #     print('label' + str(i))
    #     print(fea)
    #     print(DIVIDER)

    # seg_set = set()
    #
    # # 線分を作る
    # for num_color in trange(CLASS_SIZE, desc='create Segment'):
    #     # print(DIVIDER)
    #     # print('create Segment on label ' + str(num_color))
    #     # print(DIVIDER)
    #     color_features = color_features_list[num_color]
    #
    #     for fea1 in color_features:
    #         for fea2 in color_features:
    #
    #             if fea1[0] != fea2[0] or fea1[1] != fea2[1]:
    #                 segment = sg.Segment(sg.Point(fea1[0], fea1[1]), sg.Point(fea2[0], fea2[1]))
    #                 seg_set.add((segment, num_color))
    #
    #                 # print(DIVIDER)
    #                 # print(fea1)
    #                 # print(fea2)
    #                 # print(type(segment))
    #                 # print(segment)
    #                 # print(DIVIDER)

    # survival_seg_set = seg_set.copy()
    # survival_seg_set = seg_set

    # for seg1, seg_color1 in tqdm(seg_set, desc='check intersection'):
    #     for seg2, seg_color2 in seg_set:
    #         if seg_color1 != seg_color2:  # 色が違う
    #             result = sg.intersection(seg1, seg2)
    #             if len(result) != 0:  # 交点あり
    #                 # 線分を除外
    #                 survival_seg_set.discard(seg1)
    #                 survival_seg_set.discard(seg2)
    #
    #                 print(result)

    # # 各色の点を集めたセットを数字の個数分作る
    # color_point_set_list = []
    # for i in range(CLASS_SIZE):
    #     color_point_set_list.append(set())

    # # 点を振り分ける
    # for seg, color in survival_seg_set:
    #     point1, point2 = seg.points
    #     color_point_set_list[color].add(point1)
    #     color_point_set_list[color].add(point2)

    # 多角形作成
    polygon_set = set()
    for color in range(CLASS_SIZE):
        color_features = color_features_list[color]
        points = []
        for cnt in range(color_counts[color]):
            points.append((color_features[cnt][0], color_features[cnt][1])) # x,y

        points = set(points)
        if len(points) > 2:
            polygon = Polygon(*points)
            polygon_set.add((polygon, color))

            print(DIVIDER)
            print(type(polygon))
            print(DIVIDER)

    polygon_set_copy = polygon_set.copy()

    # # 多角形の交点を計算
    # for polygon1, color1 in polygon_set:
    #     for polygon2, color2 in polygon_set:
    #         if color1 != color2:
    #             result = sg.intersection(polygon1, polygon2)
    #             if len(result) > 0:
    #                 polygon_set_copy.discard((polygon1, color1))
    #                 polygon_set_copy.discard((polygon2, color2))

    return list(polygon_set_copy)


def draw_segments(seg_label_list, draw_segments_save_dir):
    create_dir(draw_segments_save_dir)

    for seg, label in seg_label_list:
        point1, point2 = seg.points
        x1 = float(point1.x)
        y1 = float(point1.y)
        x2 = float(point2.x)
        y2 = float(point2.y)

        print(DIVIDER)
        print('label: ' + str(label))
        print('x1: ' + str(x1))
        print('y1: ' + str(y1))
        print('x2: ' + str(x2))
        print('y2: ' + str(y2))
        print(DIVIDER)

        r, g, b = ID2COLOR[label]
        r = r / 255
        g = g / 255
        b = b / 255

        plt.plot([x1, x2], [y1, y2], color=(r, g, b))

    plt.grid(True)
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.savefig(os.path.join(draw_segments_save_dir, 'segments.png'))
    plt.show()
    plt.close()


def draw_polygons(polygon_list, save_dir):
    p = Plot(axes='label_axes=True')
    c = Circle(Point(0, 0), 1)
    p[0] = c


def create_random_xy():
    x = 2 * np.random.rand() - 1
    y = 2 * np.random.rand() - 1

    return x, y


def lv1_user_function_sampling_region(n_samples, target_model, exe_n, method_name, path_manager: SavePathManager):
    print('n_samples:' + str(n_samples))
    new_features = np.zeros((1, 2))

    if n_samples < 0:
        raise ValueError

    elif n_samples == 0:
        return np.zeros((0, 2))

    elif n_samples == 1:
        x, y = create_random_xy()
        new_features[0][0] = x
        new_features[0][1] = y
        return np.float32(new_features)

    old_features = lv1_user_function_sampling_region(n_samples=n_samples - 1, target_model=target_model,
                                                     exe_n=exe_n,
                                                     method_name=method_name,
                                                     path_manager=path_manager)

    clone_model = LV1UserDefinedClassifier()

    # target識別器からtargetのラベルを取得
    target_labels = target_model.predict(old_features)
    # clone識別器からcloneのラベルを取得
    clone_model.fit(features=old_features, labels=target_labels)
    clone_labels = clone_model.predict(features=old_features)

    polygon_list = create_region_map(features=old_features,
                                               target_labels=target_labels,
                                               clone_labels=clone_labels,
                                               n=n_samples - 1)

    point_undecided = True

    while point_undecided:
        point_undecided = False
        x, y = create_random_xy()
        new_features[0][0] = x
        new_features[0][1] = y

        for polygon, color in polygon_list:
            point = Point(x, y)
            if polygon.encloses_point(point):  # 多角形の領域外の点なら点を再決定
                point_undecided = True

    # draw_segments(polygon_list,
    #               draw_segments_save_dir=path_manager.sampling_history_n_dir(exe_n=exe_n, method_name=method_name,
    #                                                                          sampling_n=n_samples - 1))

    draw_polygons(polygon_list,
                  save_dir=path_manager.sampling_history_n_dir(exe_n=exe_n, method_name=method_name,
                                                               sampling_n=n_samples))

    return np.vstack((old_features, new_features))


def lv1_user_function_sampling_sweeper(n_samples, target_model, exe_n, method_name, path_manager: SavePathManager):
    print('n_samples:' + str(n_samples))

    if n_samples < 0:
        raise ValueError

    elif n_samples == 0:
        return np.zeros((0, 2))

    elif n_samples == 1:

        # 与えられたnによって１マスの大きさを変える
        if exe_n < 10:
            add_boat = 10
        else:
            add_boat = 0

        board_size = math.ceil(math.sqrt(exe_n)) + add_boat
        print('board_size: ' + str(board_size))
        new_board = Board(board_size=board_size)
        #new_board.init_open()

        new_features = np.zeros((1, 2))

        feature_x, feature_y = new_board.get_optimal_solution() # 最適解
        new_features[0][0] = feature_x
        new_features[0][1] = feature_y

        clone_model = LV1UserDefinedClassifier()
        # target識別器からtargetのラベルを取得
        target_labels = target_model.predict(new_features)
        # clone識別器からcloneのラベルを取得
        clone_model.fit(features=new_features, labels=target_labels)
        clone_labels = clone_model.predict(features=new_features)

        if target_labels[-1] == clone_labels[-1]:
            new_board.open_once_feature(feature_x=feature_x, feature_y=feature_y, color=clone_labels[-1]) # 開示
        else:
            new_board.open_once_feature(feature_x=feature_x, feature_y=feature_y) # 開示

        return np.float32(new_features), new_board

    elif n_samples > 1:

        old_features, old_board = lv1_user_function_sampling_sweeper(n_samples=n_samples - 1, target_model=target_model,
                                                         exe_n=exe_n,
                                                         method_name=method_name,
                                                         path_manager=path_manager)

        new_features = np.zeros((1, 2))

        feature_x, feature_y = old_board.get_optimal_solution()  # 最適解

        new_features[0][0] = feature_x
        new_features[0][1] = feature_y

        clone_model = LV1UserDefinedClassifier()
        # target識別器からtargetのラベルを取得
        target_labels = target_model.predict(new_features)
        # clone識別器からcloneのラベルを取得
        clone_model.fit(features=new_features, labels=target_labels)
        clone_labels = clone_model.predict(features=new_features)

        if target_labels[-1] == clone_labels[-1]:
            old_board.open_once_feature(feature_x=feature_x, feature_y=feature_y, color=clone_labels[-1])  # 開示
        else:
            old_board.open_once_feature(feature_x=feature_x, feature_y=feature_y)  # 開示

        return np.vstack((old_features, new_features)), old_board