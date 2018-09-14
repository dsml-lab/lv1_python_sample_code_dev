import os

from datetime import datetime

# from clone import LV1_user_function_sampling, LV1_UserDefinedClassifier, LV1_TargetClassifier
from lv2_clone import LV2_user_function_sampling, LV2_UserDefinedClassifier, LV2_TargetClassifier
from lv2_evaluation import LV2_Evaluator
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

RANDOM_SAMPLING = 'LV2_user_function_sampling'
DEMOCRACY_SAMPLING = 'democracy'


class Lv2PathManager:

    def __init__(self, save_root_path):
        self.save_root_path = save_root_path

    def sampling_method_path(self, sampling_method_name):
        return os.path.join(self.save_root_path, sampling_method_name)

    def clone_clf_path(self, sampling_method_name, clone_clf_name):
        return os.path.join(self.save_root_path, sampling_method_name, clone_clf_name)

    def target_path(self, sampling_method_name, clone_clf_name, target_name):
        return os.path.join(self.save_root_path, sampling_method_name, clone_clf_name, target_name)


def calc_diff_area(start_n, end_n, start_height, end_height):
    rect_area = (end_n - start_n) * start_height
    triangle_area = ((end_n - start_n) * (end_height - start_height)) / 2

    print(rect_area)
    print(triangle_area)

    return rect_area + triangle_area


def calc_area(n_list, evaluation_value_list):
    sum_value = 0

    for i in range(len(n_list) - 1):
        sum_value = sum_value + calc_diff_area(start_n=n_list[i],
                                               end_n=n_list[i + 1],
                                               start_height=evaluation_value_list[i],
                                               end_height=evaluation_value_list[i + 1])

    square_area = max(n_list) * 1
    ratio = sum_value / square_area
    return sum_value, ratio


def get_features(n, method_name):
    if method_name == RANDOM_SAMPLING:
        return LV2_user_function_sampling(n_samples=n)
    if method_name == DEMOCRACY_SAMPLING:
        return


def exe_my_clone(target, img_save_path, n, method_name):
    # ターゲット認識器への入力として用いる二次元特徴量を用意
    features = get_features(n, method_name)
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
    evaluator.visualize(model, img_save_path)
    print("\nThe clone recognizer was visualized and saved to {0} .".format(img_save_path))
    recall, precision, f_score = evaluator.calc_accuracy(target, model)
    print("\nrecall: {0}".format(recall))
    print("precision: {0}".format(precision))
    print("F-score: {0}".format(f_score))

    return recall, precision, f_score


def exe_my_clone_all(target_path, range_list, img_save_dir, method_name):
    n_list = []
    recall_list = []
    precision_list = []
    f_score_list = []

    # ターゲット認識器を用意
    target = LV2_TargetClassifier()
    target.load(target_path)  # 第一引数で指定された画像をターゲット認識器としてロード
    print("\nA target recognizer was loaded from {0} .".format(target_path))

    n_list.append(0)
    recall_list.append(0.0)
    precision_list.append(0.0)
    f_score_list.append(0.0)

    for n in range_list:
        save_dir = os.path.join(img_save_dir, 'n' + str(n))
        os.makedirs(save_dir)

        recall, precision, f_score = exe_my_clone(target=target,
                                                  img_save_path=save_dir,
                                                  n=n, method_name=method_name)
        n_list.append(n)
        recall_list.append(recall)
        precision_list.append(precision)
        f_score_list.append(f_score)

    return n_list, recall_list, precision_list, f_score_list


def save_csv(save_dir, n_dict, log_dict):
    df_dir = os.path.join(save_dir, '/csv/')

    save_n_accuracy_csv(n_dict=n_dict, df_dir=df_dir)
    save_log_csv(log_dict=log_dict, df_dir=df_dir)


def save_n_accuracy_csv(n_dict, df_dir):
    df = pd.DataFrame.from_dict(n_dict)
    print(df)
    df.to_csv(os.path.join(df_dir, "n_accuracy.csv"))


def save_log_csv(log_dict, df_dir):
    df = pd.DataFrame.from_dict(log_dict)
    print(df)
    df.to_csv(os.path.join(df_dir, "log.csv"))


def save_and_show_graph_f_score(n_list, f_score_list, f_score_area, method_name, graph_dir):
    left = np.asarray(n_list)
    height = np.asarray(f_score_list)
    plt.plot(left, height)
    plt.xlabel("n samples")
    plt.ylabel("F value")
    plt.grid(True)
    plt.ylim(0, 1)
    plt.title("area: " + str(f_score_area) + "\n" + method_name)
    plt.savefig(graph_dir + 'n_f_value.png')
    plt.show()


def save_and_show_graph_recall(n_list, recall_list, recall_area, method_name, graph_dir):
    left = np.asarray(n_list)
    height = np.asarray(recall_list)
    plt.plot(left, height)
    plt.xlabel("n samples")
    plt.ylabel("recall")
    plt.grid(True)
    plt.ylim(0, 1)
    plt.title("area: " + str(recall_area) + "\n" + method_name)
    plt.savefig(graph_dir + 'n_recall.png')
    plt.show()


def save_and_show_graph_precision(n_list, precision_list, precision_area, method_name, graph_dir):
    left = np.asarray(n_list)
    height = np.asarray(precision_list)
    plt.plot(left, height)
    plt.xlabel("n samples")
    plt.ylabel("precision")
    plt.grid(True)
    plt.ylim(0, 1)
    plt.title("area: " + str(precision_area) + "\n" + method_name)
    plt.savefig(graph_dir + 'n_precision.png')
    plt.show()


def create_output(save_dir):
    target_path = 'lv2_targets/classifier_01'
    method_name = RANDOM_SAMPLING
    n_list, recall_list, precision_list, f_score_list = exe_my_clone_all(target_path=target_path,
                                                                         img_save_dir=save_dir,
                                                                         range_list=range(10, 110, 10),
                                                                         method_name=method_name)

    recall_area, recall_ratio = calc_area(n_list, recall_list)
    precision_area, precision_ratio = calc_area(n_list, precision_list)
    f_score_area, f_score_ratio = calc_area(n_list, f_score_list)
    print('recallの面積: ' + str(recall_area))
    print('precisionの面積: ' + str(precision_area))
    print('f_scoreの面積: ' + str(f_score_area))

    graph_dir = os.path.join(save_dir, '/graph/')

    save_and_show_graph_recall(graph_dir=graph_dir, n_list=n_list, recall_list=recall_list, recall_area=recall_area,
                               method_name=method_name)
    save_and_show_graph_precision(graph_dir=graph_dir, n_list=n_list, precision_list=precision_list,
                                  precision_area=precision_area, method_name=method_name)
    save_and_show_graph_f_score(graph_dir=graph_dir, n_list=n_list, f_score_list=f_score_list, f_score_area=f_score_area,
                                method_name=method_name)

    n_dict = {
        'n': n_list,
        'recall': recall_list,
        'precision': precision_list,
        'f_score_list': f_score_list
    }

    log_dict = {
        'f_score_area': [f_score_area],
        'sampling_method_name': [method_name]
    }

    save_csv(save_dir=save_dir, n_dict=n_dict, log_dict=log_dict)


def run():
    now_str = datetime.now().strftime('%Y%m%d%H%M%S')
    output_path = os.path.join('output_lv2', now_str)

    create_output(output_path)


if __name__ == '__main__':
    run()