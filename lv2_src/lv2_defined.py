import numpy as np
from PIL import Image

from lv2_src.evaluation_lv2 import IMAGE_SIZE
from lv2_src.labels_lv2 import N_LABELS, ID2LNAME


class LV2TargetClassifier:

    def __init__(self, n_labels):
        self.n_labels = n_labels

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
