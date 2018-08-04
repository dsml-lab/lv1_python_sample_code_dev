import os

from datetime import datetime

from evaluation import LV1_Evaluator
from my_clone import LV1_TargetClassifier
from my_clone import LV1_user_function_sampling_meshgrid, LV1_UserDefinedClassifier
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def exe_my_clone(target, save_path, evaluator, n):
    # ターゲット認識器への入力として用いる二次元特徴量を用意
    # このサンプルコードではひとまず100サンプルを用意することにする
    features = LV1_user_function_sampling_meshgrid(n_samples=n)
    # print(features)
    #
    # print(features.shape)
    # print(features[0])
    #
    print("\n{0} features were sampled.".format(n))

    # ターゲット認識器に用意した入力特徴量を入力し，各々に対応するクラスラベルIDを取得
    labels = target.predict(features)
    # print("\nThe sampled features were recognized by the target recognizer.")

    # クローン認識器を学習
    model = LV1_UserDefinedClassifier()
    model.fit(features, labels)
    # print("\nA clone recognizer was trained.")

    # 学習したクローン認識器を可視化し，精度を評価
    evaluator.visualize(model, save_path)
    # print("\nThe clone recognizer was visualized and saved to {0} .".format(save_path))
    accuracy = evaluator.calc_accuracy(target, model)
    # print("\naccuracy: {0}".format(accuracy))

    return accuracy


def exe_my_clone_all(target_path, max_n, now_str):
    save_dir = 'output/' + now_str + '/'

    n_list = []
    acc_list = []

    # ターゲット認識器を用意
    target = LV1_TargetClassifier()
    target.load(target_path)  # 第一引数で指定された画像をターゲット認識器としてロード
    print("\nA target recognizer was loaded from {0} .".format(target_path))

    evaluator = LV1_Evaluator()

    os.makedirs(save_dir)

    for n in range(4, max_n):
        acc = exe_my_clone(target=target, evaluator=evaluator, save_path=save_dir + 'n' + str(n) + '.png', n=n)
        n_list.append(n)
        acc_list.append(acc)

    return n_list, acc_list


def show_graph():
    now_str = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    target_path = 'lv1_targets/classifier_01.png'
    n_list, acc_list = exe_my_clone_all(target_path=target_path, now_str=now_str, max_n=100)

    # n_list, acc_listをcsvで保存する処理

    left = np.array(n_list)
    height = np.array(acc_list)
    plt.bar(left, height)
    plt.xlabel("n sample count")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.show()
    plt.savefig('graph/' + now_str + '.png')


if __name__ == '__main__':
    show_graph()
