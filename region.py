import os

from datetime import datetime
import networkx as nx
import numpy as np
import sympy.geometry as sg
from sympy.geometry import Polygon, Point
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
from PIL import Image
from sklearn import neighbors

from evaluation import LV1_Evaluator
from labels import COLOR2ID, ID2COLOR

IMAGE_SIZE = 512
CLASS_SIZE = 10
DIVIDER = '------------------------'


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


def create_region_map(features, target_labels, n, clone_labels, draw_segments_save_dir):
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

    for i, fea in enumerate(color_features_list):
        print(DIVIDER)
        print('label' + str(i))
        print(fea)
        print(DIVIDER)

    seg_set = set()

    # 線分を作る
    for num_color in trange(CLASS_SIZE, desc='create Segment'):
        # print(DIVIDER)
        # print('create Segment on label ' + str(num_color))
        # print(DIVIDER)
        color_features = color_features_list[num_color]

        for fea1 in color_features:
            for fea2 in color_features:

                if fea1[0] != fea2[0] or fea1[1] != fea2[1]:
                    segment = sg.Segment(sg.Point(fea1[0], fea1[1]), sg.Point(fea2[0], fea2[1]))
                    seg_set.add((segment, num_color))

                    # print(DIVIDER)
                    # print(fea1)
                    # print(fea2)
                    # print(type(segment))
                    # print(segment)
                    # print(DIVIDER)

    survival_seg_set = seg_set.copy()

    for seg1, seg_color1 in tqdm(seg_set, desc='check intersection'):
        for seg2, seg_color2 in seg_set:
            if seg_color1 != seg_color2:  # 色が違う
                result = sg.intersection(seg1, seg2)
                if len(result) != 0:  # 交点あり
                    # 線分を除外
                    survival_seg_set.remove(seg1)
                    survival_seg_set.remove(seg2)

                    print(result)

    # seg_set_to_network(segment_set=seg_set)
    draw_segments(list(survival_seg_set), draw_segments_save_dir=draw_segments_save_dir)

    # 各色の点を集めたセットを数字の個数分作る
    color_point_set_list = []
    for i in range(CLASS_SIZE):
        color_point_set_list.append(set())

    # 点を振り分ける
    for seg, color in survival_seg_set:
        point1, point2 = seg.points
        color_point_set_list[color].add(point1)
        color_point_set_list[color].add(point2)

    polygon_list = []
    for i in range(CLASS_SIZE):
        point_list = list(color_point_set_list[i])
        points = []
        for p in point_list:
            points.append((float(p.x), float(p.y)))
        if len(points) > 2:
            polygon = Polygon(*points)
            print(type(polygon))
            polygon_list.append(polygon)

    return polygon_list


# def seg_set_to_network(segment_set):
#     graph = nx.Graph()
#
#     for segment, label in segment_set:
#         point1, point2 = segment.points
#         x1 = float(point1.x)
#         y1 = float(point1.y)
#         x2 = float(point2.x)
#         y2 = float(point2.y)
#
#         graph.add_node((x1, y1))
#         graph.add_node((x2, y2))
#
#         graph.add_edge((x1, y1), (x2, y2))
#
#     nx.draw(graph)
#     plt.show()
#     plt.close()


def draw_segments(seg_list, draw_segments_save_dir):
    # converted_seg_list = []
    for seg, label in seg_list:
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

        # converted_seg_list.append((x, y, label))

    plt.grid(True)
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.savefig(draw_segments_save_dir + 'draw_segments.png')
    plt.show()
    plt.close()


def create_random_xy():
    x = 2 * np.random.rand() - 1
    y = 2 * np.random.rand() - 1

    return x, y


def lv1_user_function_sampling_region(n_samples, target_model, draw_segments_save_dir):
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
                                                     draw_segments_save_dir=draw_segments_save_dir)

    clone_model = LV1UserDefinedClassifier()

    # target識別器からtargetのラベルを取得
    target_labels = target_model.predict(old_features)
    # clone識別器からcloneのラベルを取得
    clone_model.fit(features=old_features, labels=target_labels)
    clone_labels = clone_model.predict(features=old_features)

    polygon_list = create_region_map(features=old_features,
                                     target_labels=target_labels,
                                     clone_labels=list(clone_labels),
                                     n=n_samples-1,
                                     draw_segments_save_dir=draw_segments_save_dir)

    point_undecided = True

    while point_undecided:
        point_undecided = False
        x, y = create_random_xy()
        new_features[0][0] = x
        new_features[0][1] = y

        for polygon in polygon_list:
            point = Point(x, y)
            if polygon.encloses_point(point):  # 多角形の領域外の点なら点を再決定
                point_undecided = True

    return np.vstack((old_features, new_features))


def exe_clone(target, img_save_path, missing_img_save_path, n, draw_segments_save_dir):

    # ターゲット認識器への入力として用いる二次元特徴量を用意
    features = lv1_user_function_sampling_region(n_samples=n, target_model=target,
                                                         draw_segments_save_dir=draw_segments_save_dir)

    print(features)
    print(features.shape)
    print(features[0])
    #
    print("\n{0} features were sampled.".format(n))

    # クローン認識器を学習
    labels = target.predict(features)

    model = LV1UserDefinedClassifier()
    model.fit(features, labels)
    print("\nA clone recognizer was trained.")

    # 学習したクローン認識器を可視化し，精度を評価
    evaluator = LV1_Evaluator()
    evaluator.visualize(model, img_save_path)
    print('visualized')
    evaluator.visualize_missing(model=model, target=target, filename=missing_img_save_path, features=features)
    print("\nThe clone recognizer was visualized and saved to {0} .".format(img_save_path))
    accuracy = evaluator.calc_accuracy(target, model)
    print("\naccuracy: {0}".format(accuracy))

    return accuracy


def exe_clone_one():
    n = 30

    now_str = datetime.now().strftime('%Y%m%d%H%M%S')
    target_path = 'lv1_targets/classifier_01.png'
    draw_segments_save_dir = 'output/' + now_str + '/segments_images/'
    img_save_dir = 'output/' + now_str + '/images/'
    missing_img_save_dir = 'output/' + now_str + '/missing_images/'

    create_dir(img_save_dir)
    create_dir(missing_img_save_dir)
    create_dir(draw_segments_save_dir)

    target = LV1TargetClassifier()
    target.load(target_path)
    exe_clone(target=target,
              img_save_path=img_save_dir + 'n' + str(n) + '.png',
              missing_img_save_path=missing_img_save_dir + 'n' + str(n) + '.png',
              n=n,
              draw_segments_save_dir=draw_segments_save_dir
              )


def create_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


if __name__ == '__main__':
    exe_clone_one()
