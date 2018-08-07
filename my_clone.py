# coding: UTF-8

import sys
import numpy as np
from PIL import Image
from sklearn import neighbors
from sklearn import svm
from labels import COLOR2ID
from evaluation import IMAGE_SIZE
from evaluation import LV1_Evaluator


# ターゲット認識器を表現するクラス
# ターゲット認識器は2次元パターン（512x512の画像）で与えられるものとする
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


# クローン認識器を表現するクラス
# このサンプルコードでは単純な 1-nearest neighbor 認識器とする（sklearnを使用）
# 下記と同型の fit メソッドと predict メソッドが必要
class LV1_UserDefinedClassifier:

    def __init__(self):
        self.clf = neighbors.KNeighborsClassifier(n_neighbors=3)
        # self.clf = svm.SVC()

    # クローン認識器の学習
    #   (features, labels): 訓練データ（特徴量とラベルのペアの集合）
    def fit(self, features, labels):
        self.clf.fit(features, labels)

    # 未知の二次元特徴量を認識
    #   features: 認識対象の二次元特徴量の集合
    def predict(self, features):
        labels = self.clf.predict(features)
        return np.int32(labels)


# ターゲット認識器に入力する二次元特徴量をサンプリングする関数
#   n_samples: サンプリングする特徴量の数
def LV1_user_function_sampling(n_samples):
    features = np.zeros((n_samples, 2))
    for i in range(0, n_samples):
        # このサンプルコードでは[-1, 1]の区間をランダムサンプリングするものとする
        features[i][0] = 2 * np.random.rand() - 1
        features[i][1] = 2 * np.random.rand() - 1
    return np.float32(features)


# ターゲット認識器に入力する二次元特徴量をサンプリングする関数(格子上)
#   n_samples: サンプリングする特徴量の数
def LV1_user_function_sampling_meshgrid(n_samples):
    features = np.zeros((n_samples, 2))
    # n_samples=10 なら 3
    # n_samples=100 なら 10
    n_samples_sqrt = int(np.sqrt(n_samples))
    n_samples_sq = n_samples_sqrt * n_samples_sqrt
    # 格子ひとつ分の幅
    fragment_size = 2 / (n_samples_sqrt + 1)

    # 格子状に値を入れる
    count = 0
    for j in range(1, n_samples_sqrt + 1):
        for k in range(1, n_samples_sqrt + 1):
            features[count][0] = j * fragment_size - 1
            features[count][1] = k * fragment_size - 1
            count = count + 1

    # 残りはランダムに
    for i in range(n_samples_sq, n_samples):
        features[i][0] = 2 * np.random.rand() - 1
        features[i][1] = 2 * np.random.rand() - 1
    return np.float32(features)


# ターゲット認識器に入力する二次元特徴量をサンプリングする関数(格子・長方形)
#   n_samples: サンプリングする特徴量の数
def LV1_user_function_sampling_meshgrid_rectangular(n_samples):
    features = np.zeros((n_samples, 2))

    x_samples = 0
    y_samples = 0

    # 格子点の個数がもっとも多くなる
    # y_sizeとy_sizeの差がなるべく小さくなる

    for i in range(2, n_samples):
        for j in range(2, n_samples):
            if n_samples >= i * j > x_samples * y_samples and abs(i - j) < 5:  # nよりも小さくi*jが
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


# ターゲット認識器に入力する二次元特徴量をサンプリングする関数(適応的)
#   n_samples: サンプリングする特徴量の数
def LV1_user_function_sampling_adaptive_behavior(n_samples, target):



    target.predict_once()


def main():
    if len(sys.argv) < 3:
        print("usage: clone.py /target/classifier/image/path /output/image/path")
        exit(0)

    # ターゲット認識器を用意
    target = LV1_TargetClassifier()
    target.load(sys.argv[1])  # 第一引数で指定された画像をターゲット認識器としてロード
    print("\nA target recognizer was loaded from {0} .".format(sys.argv[1]))

    # ターゲット認識器への入力として用いる二次元特徴量を用意
    # このサンプルコードではひとまず100サンプルを用意することにする
    n = 100
    features = LV1_user_function_sampling_meshgrid(n_samples=n)
    print(features)

    print(features.shape)
    print(features[0])

    print("\n{0} features were sampled.".format(n))

    # ターゲット認識器に用意した入力特徴量を入力し，各々に対応するクラスラベルIDを取得
    labels = target.predict(features)
    print("\nThe sampled features were recognized by the target recognizer.")

    # クローン認識器を学習
    model = LV1_UserDefinedClassifier()
    model.fit(features, labels)
    print("\nA clone recognizer was trained.")

    # 学習したクローン認識器を可視化し，精度を評価
    evaluator = LV1_Evaluator()
    evaluator.visualize(model, sys.argv[2])
    print("\nThe clone recognizer was visualized and saved to {0} .".format(sys.argv[2]))
    print("\naccuracy: {0}".format(evaluator.calc_accuracy(target, model)))


# クローン処理の実行
# 第一引数でターゲット認識器を表す画像ファイルのパスを，
# 第二引数でクローン認識器の可視化結果を保存する画像ファイルのパスを，
# それぞれ指定するものとする
if __name__ == '__main__':
    main()
