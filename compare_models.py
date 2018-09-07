from PIL import Image
import numpy as np

from labels import COLOR2ID


class LV1TargetClassifier:

    def __init__(self, image_size=512):
        self.img = None
        self.image_size = image_size

    # ターゲット認識器をロード
    #   filename: ターゲット認識器を表現する画像のファイルパス
    def load(self, filename):
        self.img = Image.open(filename)

    # 入力された二次元特徴量に対し，その認識結果（クラスラベルID）を返す
    def predict_once(self, x1, x2):
        h = self.image_size // 2
        x = max(0, min(self.image_size - 1, np.round(h * x1 + h)))
        y = max(0, min(self.image_size - 1, np.round(h - h * x2)))
        return COLOR2ID(self.img.getpixel((x, y)))

    # 入力された二次元特徴量の集合に対し，各々の認識結果を返す
    def predict(self, features):
        labels = np.zeros(features.shape[0])
        for i in range(0, features.shape[0]):
            labels[i] = self.predict_once(features[i][0], features[i][1])
        return np.int32(labels)


MODEL_NAME_SVM = 'svm'
MODEL_NAME_KNN = 'knn'

TARGET_NAME_1 = 'target1'
TARGET_NAME_2 = 'target2'
TARGET_NAME_3 = 'target3'
TARGET_NAME_4 = 'target4'
TARGET_NAME_5 = 'target5'
TARGET_NAME_6 = 'target6'
TARGET_NAME_7 = 'target7'
TARGET_NAME_8 = 'target8'


class TargetModel:

    def __init__(self, created_model_name, created_target_name, target_path):
        self.created_model_name = created_model_name
        self.created_target_name = created_target_name
        self.target = LV1TargetClassifier()
        self.target.load(target_path)

    def predict(self, features):
        return self.target.predict(features=features)


