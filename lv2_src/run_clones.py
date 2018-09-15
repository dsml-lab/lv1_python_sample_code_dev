import numpy as np

# ターゲット認識器を表現するクラス
# ターゲット認識器は8枚の2次元パターン（512x512の画像）で与えられるものとする
from PIL import Image

from democ.lv2_clf import LV2UserDefinedClassifierMLP1000HiddenLayer
from democ.sampling import lv2_user_function_sampling_democracy
from lv2_src.evaluation_lv2 import LV2Evaluator
from lv2_src.labels_lv2 import N_LABELS
from lv2_src.lv2_defined import LV2TargetClassifier


def run_clone(target_path, n, visualize_directory):
    # ターゲット認識器を用意
    target = LV2TargetClassifier(n_labels=N_LABELS)
    target.load(target_path)  # ターゲット認識器としてロード

    # ターゲット認識器への入力として用いる二次元特徴量を用意
    # このサンプルコードではひとまず1000サンプルを用意することにする
    features = lv2_user_function_sampling_democracy(n_samples=n, exe_n=n, target_model=target)
    print("\n{0} features were sampled.".format(n))

    # ターゲット認識器に用意した入力特徴量を入力し，各々の認識結果（各クラスラベルの尤度を並べたベクトル）を取得
    likelihoods = target.predict_proba(features)
    print("\nThe sampled features were recognized by the target recognizer.")

    # クローン認識器を学習
    model = LV2UserDefinedClassifierMLP1000HiddenLayer(n_labels=N_LABELS)
    model.fit(features, likelihoods)
    print("\nA clone recognizer was trained.")

    # 学習したクローン認識器を可視化し，精度を評価
    evaluator = LV2Evaluator()
    evaluator.visualize(model, visualize_directory)
    recall, precision, f_score = evaluator.calc_accuracy(target, model)
    print("\nrecall: {0}".format(recall))
    print("precision: {0}".format(precision))
    print("F-score: {0}".format(f_score))

    return recall, precision, f_score

