# coding: UTF-8

import numpy as np
from PIL import Image

# ターゲット認識器を表現する画像のサイズ
from tqdm import trange

from lv2_src.labels_lv2 import ID2LNAME, N_LABELS

IMAGE_SIZE = 512


# 小数をpx値にマッピング
def mapping_x_y(feature_x, feature_y):
    h = IMAGE_SIZE // 2
    x = int(max(0, min(IMAGE_SIZE - 1, np.round(h * feature_x + h))))
    y = int(max(0, min(IMAGE_SIZE - 1, np.round(h - h * feature_y))))

    return x, y


# 構築したクローン認識器を評価するためのクラス
class LV2Evaluator:

    def __init__(self):
        h = IMAGE_SIZE // 2
        self.size = IMAGE_SIZE * IMAGE_SIZE
        self.samples = np.zeros((self.size, 2))
        for i in trange(0, self.size):
            x = i % IMAGE_SIZE
            y = i // IMAGE_SIZE
            self.samples[i][0] = np.float32((x - h) / h)
            self.samples[i][1] = np.float32(-(y - h) / h)
        self.samples = np.float32(self.samples)

    # クローン認識器を可視化する（可視化結果を8枚の画像として保存する）
    #   model: クローン認識器
    #   directory: 可視化結果の画像の保存先ディレクトリ
    def visualize(self, model, directory):
        if directory[-1] != "/" and directory[-1] != "\\":
            directory = directory + "/"
        self.clone_likelihoods = model.predict_proba(self.samples)
        for i in trange(0, N_LABELS):
            img = Image.new('L', (IMAGE_SIZE, IMAGE_SIZE))
            for j in range(0, self.size):
                x = j % IMAGE_SIZE
                y = j // IMAGE_SIZE
                img.putpixel((x, y), int(self.clone_likelihoods[j][i] * 255))
            img.save(directory + "{0}.png".format(ID2LNAME[i]))

    # ターゲット認識器とクローン認識器の出力の一致率（F値）を求める
    #   target: ターゲット認識器
    #   model: クローン認識器
    def calc_accuracy(self, target, model):
        self.target_likelihoods = target.predict_proba(self.samples)
        self.clone_likelihoods = model.predict_proba(self.samples)
        a = self.target_likelihoods >= 0.5
        b = self.clone_likelihoods >= 0.5
        c = np.logical_and(a, b)
        r_avg = 0
        p_avg = 0
        f_avg = 0
        for j in range(0, self.size):
            an = np.sum(a[j])
            bn = np.sum(b[j])
            cn = np.sum(c[j])
            if an != 0:
                r = cn / an
                r_avg += r
            if bn != 0:
                p = cn / bn
                p_avg += p
            if r != 0 or p != 0:
                f = 2 * r * p / (r + p)
                f_avg += f
        r_avg /= self.size
        p_avg /= self.size
        f_avg /= self.size
        return r_avg, p_avg, f_avg

        # クローン認識器を可視化する（可視化結果を8枚の画像として保存する）
        #   model: クローン認識器
        #   directory: 可視化結果の画像の保存先ディレクトリ
    def visualize_missing(self, model, directory, features):
        if directory[-1] != "/" and directory[-1] != "\\":
            directory = directory + "/"
        clone_likelihoods = model.predict_proba(self.samples)
        for i in trange(0, N_LABELS):
            img = Image.new('L', (IMAGE_SIZE, IMAGE_SIZE))
            for j in range(0, self.size):
                x = j % IMAGE_SIZE
                y = j // IMAGE_SIZE
                img.putpixel((x, y), int(clone_likelihoods[j][i] * 255))

            img = img.convert('RGB')

            for fea in features:
                x, y = mapping_x_y(fea[0], fea[1])
                img.putpixel((x, y), (255, 0, 0))

            img.save(directory + "{0}.png".format(ID2LNAME[i]))
