# coding: UTF-8

import sys
import numpy as np
from PIL import Image
from sklearn import neighbors
from labels import COLOR2ID
from evaluation import IMAGE_SIZE
from evaluation import LV1_Evaluator

# ターゲット認識器を表現するクラス
# ターゲット認識器は2次元パターン（512x512の画像）で与えられるものとする
from sampling import lv1_user_function_sampling_meshgrid_rectangular
from sweeper_sampling import lv1_user_function_sampling_sweeper, LV1UserDefinedClassifierSVM

# 再帰の上限
sys.setrecursionlimit(100000)

class LV1_TargetClassifier:

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

# クローン処理の実行
# 第一引数でターゲット認識器を表す画像ファイルのパスを，
# 第二引数でクローン認識器の可視化結果を保存する画像ファイルのパスを，
# それぞれ指定するものとする
if __name__ == '__main__':

    if len(sys.argv) < 3:
        print("usage: sweeper_clone.py /target/classifier/image/path /output/image/path")
        exit(0)

    # ターゲット認識器を用意
    target = LV1_TargetClassifier()
    target.load(sys.argv[1]) # 第一引数で指定された画像をターゲット認識器としてロード
    print("\nA target recognizer was loaded from {0} .".format(sys.argv[1]))

    # ターゲット認識器への入力として用いる二次元特徴量を用意
    # このサンプルコードではひとまず100サンプルを用意することにする
    n = 10000

    features = lv1_user_function_sampling_sweeper(n_samples=n, exe_n=n, target_model=target)
    # features = lv1_user_function_sampling_meshgrid_rectangular(n_samples=n)

    print("\n{0} features were sampled.".format(n))

    # ターゲット認識器に用意した入力特徴量を入力し，各々に対応するクラスラベルIDを取得
    labels = target.predict(features)
    print("\nThe sampled features were recognized by the target recognizer.")

    # クローン認識器を学習
    model = LV1UserDefinedClassifierSVM()
    model.fit(features, labels)
    print("\nA clone recognizer was trained.")

    # 学習したクローン認識器を可視化し，精度を評価
    evaluator = LV1_Evaluator()
    evaluator.visualize(model, sys.argv[2])
    print("\nThe clone recognizer was visualized and saved to {0} .".format(sys.argv[2]))
    print("\naccuracy: {0}".format(evaluator.calc_accuracy(target, model)))
