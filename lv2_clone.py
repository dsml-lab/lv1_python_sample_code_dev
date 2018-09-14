# coding: UTF-8

import sys
import numpy as np
from PIL import Image
from sklearn import neighbors
from lv2_labels import N_LABELS
from lv2_labels import ID2LNAME
from lv2_evaluation import IMAGE_SIZE
from lv2_evaluation import LV2_Evaluator

# ターゲット認識器を表現するクラス
# ターゲット認識器は8枚の2次元パターン（512x512の画像）で与えられるものとする
class LV2_TargetClassifier:

    # ターゲット認識器をロード
    #   directory: ターゲット認識器を表現する画像が置かれているディレクトリ
    def load(self, directory):
        if directory[-1] != "/" and directory[-1] != "\\":
            directory = directory + "/"
        self.imgs = []
        for i in range(0, N_LABELS):
            img = Image.open(directory + "{0}.png".format(ID2LNAME[i]))
            self.imgs.append(img)

    # 入力された二次元特徴量に対し，各クラスラベルの尤度を返す
    def predict_once(self, x1, x2):
        h = IMAGE_SIZE // 2
        x = max(0, min(IMAGE_SIZE - 1, np.round(h * x1 + h)))
        y = max(0, min(IMAGE_SIZE - 1, np.round(h - h * x2)))
        likelihood = np.zeros(N_LABELS)
        for i in range(0, N_LABELS):
            likelihood[i] = self.imgs[i].getpixel((x, y)) / 255
        return np.float32(likelihood)

    # 入力された二次元特徴量の集合に対し，各々の認識結果（全クラスラベルの尤度）を返す
    def predict_proba(self, features):
        likelihoods = []
        for i in range(0, features.shape[0]):
            l = self.predict_once(features[i][0], features[i][1])
            likelihoods.append(l)
        return np.asarray(np.float32(likelihoods))

# クローン認識器を表現するクラス
# このサンプルコードでは各クラスラベルごとに単純な 5-nearest neighbor を行うものとする（sklearnを使用）
# 下記と同型の fit メソッドと predict_proba メソッドが必要
class LV2_UserDefinedClassifier:

    def __init__(self):
        self.clfs = []
        for i in range(0, N_LABELS):
            clf = neighbors.KNeighborsClassifier(n_neighbors=5)
            self.clfs.append(clf)

    # クローン認識器の学習
    #   (features, likelihoods): 訓練データ（特徴量と尤度ベクトルのペアの集合）
    def fit(self, features, likelihoods):
        labels = np.int32(likelihoods >= 0.5) # 尤度0.5以上のラベルのみがターゲット認識器の認識結果であると解釈する
        for i in range(0, N_LABELS):
            l = labels[:,i]
            self.clfs[i].fit(features, l)

    # 未知の二次元特徴量を認識
    #   features: 認識対象の二次元特徴量の集合
    def predict_proba(self, features):
        likelihoods = np.c_[np.zeros(features.shape[0])]
        for i in range(0, N_LABELS):
            p = self.clfs[i].predict_proba(features)
            likelihoods = np.hstack([likelihoods, np.c_[p[:,1]]])
        likelihoods = likelihoods[:, 1:]
        return np.float32(likelihoods)

# ターゲット認識器に入力する二次元特徴量をサンプリングする関数
#   n_samples: サンプリングする特徴量の数
def LV2_user_function_sampling(n_samples=1):
    features = np.zeros((n_samples, 2))
    for i in range(0, n_samples):
        # このサンプルコードでは[-1, 1]の区間をランダムサンプリングするものとする
        features[i][0] = 2 * np.random.rand() - 1
        features[i][1] = 2 * np.random.rand() - 1
    return np.float32(features)


def main():
    if len(sys.argv) < 3:
        print("usage: lv2_clone.py /target/classifier/directory/path /output_lv2/directory/path")
        exit(0)

    # ターゲット認識器を用意
    target = LV2_TargetClassifier()
    target.load(sys.argv[1]) # 第一引数で指定されたディレクトリ内の画像をターゲット認識器としてロード
    print("\nA target recognizer was loaded from {0} .".format(sys.argv[1]))

    # ターゲット認識器への入力として用いる二次元特徴量を用意
    # このサンプルコードではひとまず1000サンプルを用意することにする
    n = 1000
    features = LV2_user_function_sampling(n_samples=n)
    print("\n{0} features were sampled.".format(n))

    # ターゲット認識器に用意した入力特徴量を入力し，各々の認識結果（各クラスラベルの尤度を並べたベクトル）を取得
    likelihoods = target.predict_proba(features)
    print("\nThe sampled features were recognized by the target recognizer.")

    # クローン認識器を学習
    model = LV2_UserDefinedClassifier()
    model.fit(features, likelihoods)
    print("\nA clone recognizer was trained.")

    # 学習したクローン認識器を可視化し，精度を評価
    evaluator = LV2_Evaluator()
    evaluator.visualize(model, sys.argv[2])
    print("\nThe clone recognizer was visualized and saved to {0} .".format(sys.argv[2]))
    recall, precision, f_score =evaluator.calc_accuracy(target, model)
    print("\nrecall: {0}".format(recall))
    print("precision: {0}".format(precision))
    print("F-score: {0}".format(f_score))


# クローン処理の実行
# 第一引数でターゲット認識器を表す画像ファイルが格納されているディレクトリを
# 第二引数でクローン認識器の可視化結果の保存先ディレクトリを，
# それぞれ指定するものとする
if __name__ == '__main__':
    main()