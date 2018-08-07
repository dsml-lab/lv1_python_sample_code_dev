# coding: UTF-8

import numpy as np
from PIL import Image
from labels import ID2COLOR

# ターゲット認識器を表現する画像のサイズ
IMAGE_SIZE = 512


# 小数をpx値にマッピング
def mapping_x_y(feature_x, feature_y):
    h = IMAGE_SIZE // 2
    x = int(max(0, min(IMAGE_SIZE - 1, np.round(h * feature_x + h))))
    y = int(max(0, min(IMAGE_SIZE - 1, np.round(h - h * feature_y))))

    return x, y


# pxの座標がサンプリングした点である
def is_sampling_position(x, y, features):
    # features[count][0] xの点
    # features[count][1] yの点
    for fea in features:
        fea_x, fea_y = mapping_x_y(fea[0], fea[1])

        if fea_x == x and fea_y == y:
            # print('fea_x:' + str(fea_x))
            # print('fea_y:' + str(fea_y))
            # print('x:' + str(x))
            # print('y:' + str(y))
            return True

    return False


# 構築したクローン認識器を評価するためのクラス
class LV1_Evaluator:

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
        for i in range(0, self.size):
            x = i % IMAGE_SIZE
            y = i // IMAGE_SIZE
            img.putpixel((x, y), ID2COLOR[self.clone_labels[i]])
        img.save(filename)

    # ターゲット認識器とクローン認識器の出力の一致率を求める
    #   target: ターゲット認識器
    #   model: クローン認識器
    def calc_accuracy(self, target, model):
        self.target_labels = target.predict(self.samples)
        self.clone_labels = model.predict(self.samples)
        n = np.count_nonzero(self.target_labels - self.clone_labels)
        return (self.size - n) / self.size

    # クローン認識器の誤り部分を可視化する（可視化結果を画像として保存する）
    #   model: クローン認識器
    #   filename: 可視化結果の保存先画像のファイルパス
    def visualize_missing(self, target, model, filename, features):
        self.target_labels = target.predict(self.samples)
        self.clone_labels = model.predict(self.samples)
        img = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE))
        for i in range(0, self.size):
            x = i % IMAGE_SIZE
            y = i // IMAGE_SIZE

            if is_sampling_position(x=x, y=y, features=features):
                img.putpixel((x, y), (255, 255, 255))
            elif self.target_labels[i] - self.clone_labels[i] == 0:  # targetとcloneが一致
                img.putpixel((x, y), ID2COLOR[self.clone_labels[i]])
            else:
                rgb = list(ID2COLOR[self.clone_labels[i]])
                rgb = map(lambda c: int(c / 2), rgb)
                rgb = tuple(rgb)
                img.putpixel((x, y), rgb)
        img.save(filename)
