from datetime import datetime

import numpy as np
import sympy.geometry as sg
import matplotlib.pyplot as plt
from PIL import Image
from sklearn import neighbors

from evaluation import LV1_Evaluator
from labels import COLOR2ID

IMAGE_SIZE = 512
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
    color_counts = np.zeros(10, dtype=np.int)

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
    for i in range(10):
        color_features_list.append(np.zeros((color_counts[i], 2)))

    # 各々の色の配列に値を代入
    for num_color in range(10):
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
    for num_color in range(10):
        color_features = color_features_list[num_color]

        for fea1 in color_features:
            for fea2 in color_features:

                if fea1[0] != fea2[0] or fea1[1] != fea2[1]:
                    segment = sg.Segment(sg.Point(fea1[0], fea1[1]), sg.Point(fea2[0], fea2[1]))
                    seg_set.add((segment, num_color))

                    print(DIVIDER)
                    print('fea1')
                    print(fea1)
                    print('fea2')
                    print(fea2)
                    print(type(segment))
                    print(segment)
                    print(DIVIDER)

    survival_seg_set = seg_set.copy()

    for seg1, seg_color1 in seg_set:
        for seg2, seg_color2 in seg_set:

            if seg_color1 != seg_color2:  # 色が違う
                result = sg.intersection(seg1, seg2)
                if len(result) != 0:  # 交点あり
                    # 線分を除外
                    survival_seg_set.remove(seg1)
                    survival_seg_set.remove(seg2)

    draw_segments(list(survival_seg_set))


def draw_segments(seg_list):
    # converted_seg_list = []
    for seg, label in seg_list:
        print(type(seg))
        point1, point2 = seg.points
        x1 = float(point1.x)
        y1 = float(point1.y)
        x2 = float(point2.x)
        y2 = float(point2.y)

        plt.plot([x1, y1], [x2, y2], 'b-o')
        # converted_seg_list.append((x, y, label))

    plt.savefig('draw_segments.png')
    plt.show()
    plt.close()


def lv1_user_function_sampling_region(n_samples, target_model, clone_model):
    print('n_samples:' + str(n_samples))

    new_features = np.zeros((n_samples, 2))
    for i in range(0, n_samples):
        # このサンプルコードでは[-1, 1]の区間をランダムサンプリングするものとする
        new_features[i][0] = 2 * np.random.rand() - 1
        new_features[i][1] = 2 * np.random.rand() - 1

    # target識別器からtargetのラベルを取得
    target_labels = target_model.predict(new_features)
    # clone識別器からcloneのラベルを取得
    clone_model.fit(features=new_features, labels=target_labels)
    clone_labels = clone_model.predict(features=new_features)

    create_region_map(features=new_features, target_labels=target_labels, clone_labels=clone_labels, n=n_samples)

    return np.float32(new_features), target_labels

    if n_samples == 1:
        return np.float32(new_features)

    # old_features = lv1_user_function_sampling_region(n_samples=n_samples - 1, features=)

    return np.vstack((old_features, new_features))


def exe_clone(target, img_save_path, missing_img_save_path, n):
    model = LV1UserDefinedClassifier()

    # ターゲット認識器への入力として用いる二次元特徴量を用意
    features, labels = lv1_user_function_sampling_region(n_samples=n, target_model=target, clone_model=model)

    print(features)

    print(features.shape)
    print(features[0])
    #
    print("\n{0} features were sampled.".format(n))

    # クローン認識器を学習
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
    n = 6

    now_str = datetime.now().strftime('%Y%m%d%H%M%S')
    target_path = 'lv1_targets/classifier_01.png'
    img_save_dir = 'output/' + now_str + '/images/'
    missing_img_save_dir = 'output/' + now_str + '/missing_images/'

    target = LV1TargetClassifier()
    target.load(target_path)
    exe_clone(target=target,
              img_save_path=img_save_dir + 'n' + str(n) + '.png',
              missing_img_save_path=missing_img_save_dir + 'n' + str(n) + '.png',
              n=n)


if __name__ == '__main__':
    exe_clone_one()
