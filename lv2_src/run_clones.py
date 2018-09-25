import os

from democ.lv2_clf import LV2UserDefinedClassifierMLP1000HiddenLayer, LV2UserDefinedClassifierKerasMLP
from democ.sampling import lv2_user_function_sampling_democracy
from lv2_src.caluculator import calc_area, area_statistics, save_area_text
from lv2_src.evaluation_lv2 import LV2Evaluator
from lv2_src.labels_lv2 import N_LABELS
from lv2_src.lv2_defined import LV2TargetClassifier
# ターゲット認識器を表現するクラス
# ターゲット認識器は8枚の2次元パターン（512x512の画像）で与えられるものとする
from lv2_src.path_manage import load_directories, get_root_dir, create_dir
import numpy as np


# ターゲット認識器に入力する二次元特徴量をサンプリングする関数
#   n_samples: サンプリングする特徴量の数
def LV2_user_function_sampling(n_samples=1):
    features = np.zeros((n_samples, 2))
    for i in range(0, n_samples):
        # このサンプルコードでは[-1, 1]の区間をランダムサンプリングするものとする
        features[i][0] = 2 * np.random.rand() - 1
        features[i][1] = 2 * np.random.rand() - 1
    return np.float32(features)


def run_clone(target_path, n, visualize_directory):
    # ターゲット認識器を用意
    target = LV2TargetClassifier(n_labels=N_LABELS)
    target.load(target_path)  # ターゲット認識器としてロード

    # ターゲット認識器への入力として用いる二次元特徴量を用意
    # このサンプルコードではひとまず1000サンプルを用意することにする
    features, likelihoods = lv2_user_function_sampling_democracy(n_samples=n, exe_n=n, target_model=target)
    print("\n{0} features were sampled.".format(n))

    print("\nThe sampled features were recognized by the target recognizer.")

    # クローン認識器を学習
    model = LV2UserDefinedClassifierKerasMLP(n_labels=N_LABELS)
    model.fit(features, likelihoods)
    print("\nA clone recognizer was trained.")

    create_dir(visualize_directory)

    # 学習したクローン認識器を可視化し，精度を評価
    evaluator = LV2Evaluator()
    # evaluator.visualize(model, visualize_directory)
    evaluator.visualize_missing(model, visualize_directory, features)
    print('visualized')
    recall, precision, f_score = evaluator.calc_accuracy(target, model)
    print("\nrecall: {0}".format(recall))
    print("precision: {0}".format(precision))
    print("F-score: {0}".format(f_score))

    return recall, precision, f_score


def run_clone_random(target_path, n, visualize_directory, model):
    # ターゲット認識器を用意
    target = LV2TargetClassifier(n_labels=N_LABELS)
    target.load(target_path)  # ターゲット認識器としてロード

    # ターゲット認識器への入力として用いる二次元特徴量を用意
    # このサンプルコードではひとまず1000サンプルを用意することにする
    features = LV2_user_function_sampling(n_samples=n)
    print("\n{0} features were sampled.".format(n))

    # ターゲット認識器に用意した入力特徴量を入力し，各々に対応するクラスラベルIDを取得
    likelihoods = target.predict_proba(features)
    print("\n{0} features were sampled.".format(n))

    print("\nThe sampled features were recognized by the target recognizer.")

    # クローン認識器を学習

    model.fit(features, likelihoods)
    print("\nA clone recognizer was trained.")

    create_dir(visualize_directory)

    # 学習したクローン認識器を可視化し，精度を評価
    evaluator = LV2Evaluator()
    # evaluator.visualize(model, visualize_directory)
    evaluator.visualize_missing(model, visualize_directory, features)
    print('visualized')
    recall, precision, f_score = evaluator.calc_accuracy(target, model)
    print("\nrecall: {0}".format(recall))
    print("precision: {0}".format(precision))
    print("F-score: {0}".format(f_score))

    return recall, precision, f_score


def run_clone_one_mlp1000():
    n = 100
    model = LV2UserDefinedClassifierMLP1000HiddenLayer(n_labels=N_LABELS)
    run_clone_random(target_path='lv2_targets/classifier_04',
                     n=n,
                     visualize_directory=os.path.join(get_root_dir() + '_mlp', str('mlp1000')),
                     model=model)


def run_clone_one_mlp_keras():
    n = 100
    model = LV2UserDefinedClassifierKerasMLP(n_labels=N_LABELS)
    run_clone_random(target_path='lv2_targets/classifier_04',
                     n=n,
                     visualize_directory=os.path.join(get_root_dir(), str('keras')),
                     model=model)


def run_clone_area(target_path, save_area_path, n_list):
    recall_list = []
    precision_list = []
    f_score_list = []
    for n in n_list:
        recall, precision, f_score = run_clone(n=n, target_path=target_path,
                                               visualize_directory=os.path.join(save_area_path, str(n)))

        recall_list.append(recall)
        precision_list.append(precision)
        f_score_list.append(f_score)

    f_score_area = calc_area(n_list=n_list, acc_list=f_score_list)
    ratio = f_score_area / max(n_list)

    save_area_text(
        save_path=os.path.join(save_area_path, 'area_text.png'),
        area=f_score_area,
        area_size_x=max(n_list),
        area_size_y=1
    )

    return f_score_area, ratio


def run_clone_area_each_targets(targets_path, save_parent_path):
    target_path_list = load_directories(path=targets_path)
    n_list = range(10, 110, 10)
    target_names = []
    f_score_area_list = []
    ratio_list = []

    for path in target_path_list:
        f_score_area, ratio = run_clone_area(target_path=path,
                                             save_area_path=os.path.join(save_parent_path, os.path.basename(path)),
                                             n_list=n_list)
        target_names.append(os.path.basename(path))
        f_score_area_list.append(f_score_area)
        ratio_list.append(ratio)

    print(ratio_list)

    area_statistics(save_path=os.path.join(save_parent_path, 'statistics.png'),
                    areas=f_score_area_list,
                    target_names=target_names,
                    x_size=max(n_list),
                    y_size=1,
                    title='democracy'
                    )


def run():
    run_clone_area_each_targets(targets_path='lv2_targets', save_parent_path=get_root_dir())
