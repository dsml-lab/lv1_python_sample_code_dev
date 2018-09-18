# coding: UTF-8

import numpy as np
from PIL import Image
from tqdm import trange

# ターゲット認識器を表現する画像のサイズ
from lv1_src.labels import ID2COLOR

IMAGE_SIZE = 512
WHITE = (255, 255, 255)


def to_decimal_from_px(px):
    return (px / (IMAGE_SIZE - 1)) * 2 - 1


# 小数をpx値にマッピング
def mapping_x_y(feature_x, feature_y):
    h = IMAGE_SIZE // 2
    x = int(max(0, min(IMAGE_SIZE - 1, np.round(h * feature_x + h))))
    y = int(max(0, min(IMAGE_SIZE - 1, np.round(h - h * feature_y))))

    return x, y


# 構築したクローン認識器を評価するためのクラス
class LV1Evaluator:

    def __init__(self):
        h = IMAGE_SIZE // 2
        self.size = IMAGE_SIZE * IMAGE_SIZE
        self.samples = np.zeros((self.size, 2))
        for i in range(0, self.size):
            x = i % IMAGE_SIZE
            y = i // IMAGE_SIZE
            self.samples[i][0] = np.float32((x - h) / h)
            self.samples[i][1] = np.float32(-(y - h) / h)
        self.samples = np.float32(self.samples)

    # クローン認識器を可視化する（可視化結果を画像として保存する）
    #   model: クローン認識器
    #   filename: 可視化結果の保存先画像のファイルパス
    def visualize(self, model, filename):
        self.clone_labels = model.predict(self.samples)
        img = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE))
        for i in trange(0, self.size, desc='visualize'):
            x = i % IMAGE_SIZE
            y = i // IMAGE_SIZE
            img.putpixel((x, y), ID2COLOR[self.clone_labels[i]])
        img.save(filename)

    # クローン認識器を可視化する（可視化結果を画像として保存する）
    #   model: クローン認識器
    #   filename: 可視化結果の保存先画像のファイルパス
    def visualize_get_img(self, model):
        clone_labels = model.predict(self.samples)
        img = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE))
        for i in range(0, self.size):
            x = i % IMAGE_SIZE
            y = i // IMAGE_SIZE
            img.putpixel((x, y), ID2COLOR[clone_labels[i]])
        return img

    # クローン認識器を可視化する（可視化結果を画像として保存する）
    #   model: クローン認識器
    #   filename: 可視化結果の保存先画像のファイルパス
    def edge_img_to_edge_features(self, edge_img, edge_distance):

        xy_list = []
        for i in range(0, self.size):
            x = i % IMAGE_SIZE
            y = i // IMAGE_SIZE
            if WHITE == edge_img.getpixel((x, y)) and x > 0 and y > 0 and x < (IMAGE_SIZE - 1) and y < (IMAGE_SIZE - 1):
                xy_list.append((x, y))

        features = np.zeros((len(xy_list), 2))
        for i in range(len(xy_list)):
            x, y = xy_list[i]
            features[i][0] = to_decimal_from_px(x)
            features[i][1] = -to_decimal_from_px(y)

        features_gap_up = np.zeros((len(xy_list), 2))
        for i in range(len(xy_list)):
            x, y = xy_list[i]
            features_gap_up[i][0] = to_decimal_from_px(x)
            features_gap_up[i][1] = max(-to_decimal_from_px(y) - edge_distance, -1)

        features_gap_down = np.zeros((len(xy_list), 2))
        for i in range(len(xy_list)):
            x, y = xy_list[i]
            features_gap_down[i][0] = to_decimal_from_px(x)
            features_gap_down[i][1] = min(-to_decimal_from_px(y) + edge_distance, 1)

        features_gap_left = np.zeros((len(xy_list), 2))
        for i in range(len(xy_list)):
            x, y = xy_list[i]
            features_gap_left[i][0] = max(to_decimal_from_px(x) - edge_distance, -1)
            features_gap_left[i][1] = -to_decimal_from_px(y)

        features_gap_right = np.zeros((len(xy_list), 2))
        for i in range(len(xy_list)):
            x, y = xy_list[i]
            features_gap_right[i][0] = min(to_decimal_from_px(x) + edge_distance, 1)
            features_gap_right[i][1] = -to_decimal_from_px(y)

        features_gaps_vertical = np.vstack((features_gap_up, features_gap_down))
        features_gaps_horizontal = np.vstack((features_gap_left, features_gap_right))
        features_gaps = np.vstack((features_gaps_vertical, features_gaps_horizontal))
        features = np.vstack((features, features_gaps))

        return np.float32(features)

    # ターゲット認識器とクローン認識器の出力の一致率を求める
    #   target: ターゲット認識器
    #   model: クローン認識器
    def calc_accuracy(self, target, model):
        self.target_labels = target.predict(self.samples)
        self.clone_labels = model.predict(self.samples)
        n = np.count_nonzero(self.target_labels - self.clone_labels)
        return (self.size - n) / self.size

    def calc_sampling_accuracy(self, sampling_features, target, model):
        # サンプリング点に関して、accuracyを求める
        target_labels = target.predict(sampling_features)
        clone_labels = model.predict(sampling_features)
        n = np.count_nonzero(target_labels - clone_labels)
        return (self.size - n) / self.size

    # クローン認識器の誤り部分を可視化する（可視化結果を画像として保存する）
    #   model: クローン認識器
    #   filename: 可視化結果の保存先画像のファイルパス
    def visualize_missing(self, target, model, filename, features):
        self.target_labels = target.predict(self.samples)
        self.clone_labels = model.predict(self.samples)
        img = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE))
        for i in trange(0, self.size, desc='missing visualize'):
            x = i % IMAGE_SIZE
            y = i // IMAGE_SIZE

            if self.target_labels[i] - self.clone_labels[i] == 0:  # targetとcloneが一致
                img.putpixel((x, y), ID2COLOR[self.clone_labels[i]])
            else:
                rgb = list(ID2COLOR[self.clone_labels[i]])
                rgb = map(lambda c: int(c / 2), rgb)
                rgb = tuple(rgb)
                img.putpixel((x, y), rgb)

        for fea in features:
            x, y = mapping_x_y(fea[0], fea[1])
            img.putpixel((x, y), (255, 255, 255))

        img.save(filename)
