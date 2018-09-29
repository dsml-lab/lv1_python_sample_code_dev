# coding: UTF-8

import csv
import os
from datetime import datetime
import pandas as pd
import git

import numpy as np
from PIL import Image

# ラベルリストのファイルパス
# ダウンロード先に応じて適宜変更してください

from tqdm import trange

from democ.lv3_clf import LV3UserDefinedClassifier, vgg_input_value, LV3UserDefinedClassifierDivide, \
    LV3UserDefinedClassifierAllOne
from democ.sampling import lv3_user_function_sampling_democracy, lv3_user_function_sampling_democracy_ecology
from lv3_src.evaluation import LV3_Evaluator
from lv3_src.extractor import LV3FeatureExtractor
from lv3_src.labels import LabelTable

import sys

sys.setrecursionlimit(10000)

LABEL_LIST = "lv3_src/lv3_label_list.csv"

# データセットが存在するディレクトリのパス
# ダウンロード・解凍先に応じて適宜変更してください
# DATASET_PATH = "/media/kiyo/SDPX-USC/lv3_dataset/"
DATASET_PATH = "../../lv3_dataset/"

# クローン認識器訓練用画像が存在するディレクトリのパス
TRAIN_IMAGE_DIR = DATASET_PATH + "train/"

# クローン認識器評価用画像が存在するディレクトリのパス
VALID_IMAGE_DIR = DATASET_PATH + "valid/"

# ラベル表： ラベル名とラベルIDを相互に変換するための表
# グローバル変数として定義
LT = LabelTable(LABEL_LIST)


# ターゲット認識器への入力対象となる画像データセットを表すクラス
class LV3_ImageSet:

    # 画像ファイル名のリストを読み込む
    def __init__(self, image_dir):
        self.imgfiles = []
        f = open(image_dir + "image_list.csv", "r")
        for line in f:
            filename = line.rstrip()  # 改行文字を削除
            self.imgfiles.append(image_dir + filename)
        f.close()

    # データセットサイズ（画像の枚数）
    def size(self):
        return len(self.imgfiles)

    # n番目の画像を取得
    #   as_gray: Trueなら1-channel画像，Falseなら3-channels画像として読み込む
    def get_image(self, n, as_gray=False):
        if as_gray == True:
            img = Image.open(self.imgfiles[n]).convert("L")
        else:
            img = Image.open(self.imgfiles[n]).convert("RGB")
        img = img.resize((vgg_input_value, vgg_input_value), Image.BILINEAR)  # 処理時間短縮のため画像サイズを128x128に縮小
        return np.asarray(img, dtype=np.uint8)

    # n番目の画像の特徴量を取得
    #   extractor: LV3_FeatureExtractorクラスのインスタンス
    def get_feature(self, n, extractor):
        img = self.get_image(n, as_gray=True)
        return extractor.extract(img=img)

    # n番目の画像の特徴量を取得
    #   extractor: LV3_FeatureExtractorクラスのインスタンス
    def get_feature_lbp(self, n, extractor):
        img = self.get_image(n, as_gray=True)
        return extractor.extract(img=img)


# ターゲット認識器を表現するクラス
# ターゲット認識器は画像ID（整数値）とそのクラスラベル（マルチラベル，尤度つき）のリストで与えられるものとする
class LV3_TargetClassifier:

    # ターゲット認識器をロード
    #   filename: ターゲット認識器を表すリストファイルのパス
    def load(self, filename):
        global LT
        self.labels = []
        self.likelihoods = []
        f = open(filename, "r")
        reader = csv.reader(f)
        for row in reader:
            temp_label = []
            temp_likelihood = []
            for i in range(1, len(row), 2):
                temp_label.append(LT.LNAME2ID(row[i]))
                temp_likelihood.append(float(row[i + 1]))
            self.labels.append(np.asarray(temp_label, dtype=np.int32))
            self.likelihoods.append(np.asarray(temp_likelihood, dtype=np.float32))
        f.close()

    # ターゲット認識器として使用中の画像リストのサイズ
    def size(self):
        return len(self.labels)

    # 単一サンプルに対し，各クラスラベルの尤度を返す
    #   feature: 関数LV3_user_function_sampling()でサンプリングした特徴量の一つ一つ
    def predict_once(self, feature):
        global LT
        n = feature[0]
        likelihood = np.zeros(LT.N_LABELS())
        for i in range(0, self.labels[n].shape[0]):
            likelihood[self.labels[n][i]] = self.likelihoods[n][i]
        return np.float32(likelihood)

    # 複数サンプルに対し，各クラスラベルの尤度を返す
    #   features: 関数LV3_user_function_sampling()でサンプリングした特徴量
    def predict_proba(self, features):
        likelihoods = []
        for i in range(0, len(features)):
            l = self.predict_once(features[i])
            likelihoods.append(l)
        return np.asarray(likelihoods, dtype=np.float32)


# ターゲット認識器に入力する画像特徴量をサンプリングする関数
#   set: LV3_ImageSetクラスのインスタンス
#   extractor: LV3_FeatureExtractorクラスのインスタンス
#   n_samples: サンプリングする特徴量の数
def LV3_user_function_sampling(set, extractor, n_samples=1):
    all_image_num = 50000

    # まず，画像データセット中の全画像から特徴量を抽出する
    # 本サンプルコードでは処理時間短縮のため先頭5,000枚のみを対象とする
    # 不要なら行わなくても良い
    all_features = []
    for i in trange(0, all_image_num, desc='5000 load'):
        f = set.get_feature(i, extractor)
        all_features.append((i, f))  # 画像番号と特徴量の組を保存

    # 特徴量の集合からn_samples個をランダムに抽出する
    perm = np.random.permutation(all_image_num)
    features = []
    for i in trange(0, n_samples, desc='2000 select'):
        features.append(all_features[perm[i]])

    return features


# クローン処理の実行
# 第一引数でターゲット認識器を表す画像ファイルが格納されているディレクトリを指定するものとする
if __name__ == '__main__':
    print(LT.labels)
    print(LT.N_LABELS())
    # if len(sys.argv) < 2:
    #     print("usage: clone.py /target/classifier/path")
    #     exit(0)

    # 訓練用画像データセットをロード
    train_set = LV3_ImageSet(TRAIN_IMAGE_DIR)
    print("\nAn image dataset for training a clone recognizer was loaded.")

    # 特徴量抽出器を用意
    extractor = LV3FeatureExtractor()

    # ターゲット認識器を用意
    # target_dir = sys.argv[1]
    target_dir = 'lv3_targets/classifier_01'
    if target_dir[-1] != "/" and target_dir[-1] != "\\":
        target_dir = target_dir + "/"
    target = LV3_TargetClassifier()
    target.load(target_dir + "train.csv")  # ターゲット認識器をロード
    # print("\nA target recognizer was loaded from {0} .".format(sys.argv[1]))

    # ターゲット認識器への入力として用いる特徴量を用意
    # このサンプルコードではひとまず2,000サンプルを用意することにする
    n = 10000
    features = lv3_user_function_sampling_democracy(data_set=train_set,
                                                            extractor=extractor,
                                                            n_samples=n,
                                                            target_model=target,
                                                            labels_all=LT.labels,
                                                            all_image_num=100000
                                                            )
    # features = LV3_user_function_sampling(set=train_set, extractor=extractor, n_samples=n)
    print("\n{0} features were sampled.".format(n))

    # ターゲット認識器に用意した入力特徴量を入力し，各々の認識結果（各クラスラベルの尤度を並べたベクトル）を取得
    likelihoods = target.predict_proba(features)
    print("\nThe sampled features were recognized by the target recognizer.")

    # クローン認識器を学習
    model = LV3UserDefinedClassifierDivide(labels_all=LT.labels)
    # model = LV3UserDefinedClassifier(n_labels=LT.N_LABELS())
    model.fit(features, likelihoods)
    print("\nA clone recognizer was trained.")

    # 学習したクローン認識器の精度を評価
    valid_set = LV3_ImageSet(VALID_IMAGE_DIR)  # 評価用画像データセットをロード
    evaluator = LV3_Evaluator(valid_set, extractor)
    target.load(target_dir + "valid.csv")
    recall, precision, f_score = evaluator.calc_accuracy(target, model)
    print("\nrecall: {0}".format(recall))
    print("precision: {0}".format(precision))
    print("F-score: {0}".format(f_score))

    now = datetime.now().strftime('%Y%m%d%H%M%S')
    save_dir = 'output_lv3/' + now
    os.makedirs(save_dir)

    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    commit_hash = repo.git.rev_parse(sha)

    dic = {'n': n,
           'commit_hash': commit_hash,
           'recall': recall,
           'precision': precision,
           'F-score': f_score}
    df = pd.DataFrame(dic, index=['i', ])
    df.to_csv(os.path.join(save_dir, 'f_value.csv'))
