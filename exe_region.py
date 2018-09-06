import math

import os

from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

from democracy import lv1_user_function_sampling_democracy, LV1UserDefinedClassifierSVM10C10GammaGridSearch, \
    LV1UserDefinedClassifierTree1000MaxDepth, LV1UserDefinedClassifierRandomForest, \
    LV1UserDefinedClassifierMLP1000HiddenLayer, LV1UserDefinedClassifier1NN, \
    lv1_user_function_sampling_democracy_and_grid, LV1UserDefinedClassifier1NNRetry
from evaluation import LV1_Evaluator
from region import SavePathManager, create_dir, LV1TargetClassifier, DIVIDER, LV1UserDefinedClassifier
from sampling import lv1_user_function_sampling
from sweeper_sampling import lv1_user_function_sampling_sweeper, LV1UserDefinedClassifierSVM, \
    lv1_user_function_sampling_sweeper_colorless, lv1_user_function_sampling_meshgrid_rectangular, \
    lv1_user_function_sampling_sweeper_pixel, lv1_user_function_sampling_sweeper_start

METHOD_NAME_REGION = 'lv1_user_function_sampling_region'
METHOD_NAME_SWEEPER = 'lv1_user_function_sampling_sweeper'
METHOD_NAME_SWEEPER_pixel = 'lv1_user_function_sampling_sweeper_pixel'
METHOD_NAME_SWEEPER_COLORLESS = 'lv1_user_function_sampling_sweeper_colorless'
METHOD_NAME_GRID = 'lv1_user_function_sampling_meshgrid_rectangular'
METHOD_NAME_RANDOM = 'lv1_user_function_sampling'
METHOD_NAME_OR = 'lv1_user_function_sampling_sweeper_or_grid_or_grid_edge'
METHOD_NAME_democracy = 'lv1_user_function_sampling_democracy'
METHOD_NAME_democracy_demo = 'lv1_user_function_sampling_democracy_demo'

area_pixel = []


def get_features(target, exe_n,
                 method_name
                 ):
    if method_name == METHOD_NAME_SWEEPER_COLORLESS:
        return lv1_user_function_sampling_sweeper_colorless(n_samples=exe_n, target_model=target,exe_n=exe_n, board_size_x=math.ceil(math.sqrt(exe_n)) + 100, board_size_y=math.ceil(math.sqrt(exe_n)) + 100)

    if method_name == METHOD_NAME_SWEEPER:
        return lv1_user_function_sampling_sweeper(n_samples=exe_n, target_model=target, exe_n=exe_n, board_size_x=math.ceil(math.sqrt(exe_n)) + 100, board_size_y=math.ceil(math.sqrt(exe_n)) + 100)

    if method_name == METHOD_NAME_SWEEPER_pixel:
        return lv1_user_function_sampling_sweeper_pixel(n_samples=exe_n, target_model=target, exe_n=exe_n)

    if method_name == METHOD_NAME_GRID:
        return lv1_user_function_sampling_meshgrid_rectangular(n_samples=exe_n)

    if method_name == METHOD_NAME_RANDOM:
        return lv1_user_function_sampling(n_samples=exe_n)

    if method_name == METHOD_NAME_OR:
        return lv1_user_function_sampling_sweeper_start(n_samples=exe_n, target_model=target)

    if method_name == METHOD_NAME_democracy:
        return lv1_user_function_sampling_democracy(n_samples=exe_n, target_model=target, exe_n=exe_n)

    if method_name == METHOD_NAME_democracy_demo:
        return  lv1_user_function_sampling_democracy_and_grid(n_samples=exe_n, target_model=target)


def exe_clone(target, exe_n, method_name, path_manager: SavePathManager):
    # ターゲット認識器への入力として用いる二次元特徴量を用意
    features = get_features(target=target, exe_n=exe_n, method_name=method_name)

    print(features)
    print(features.shape)
    print(features[0])
    #
    print("\n{0} features were sampled.".format(exe_n))

    # クローン認識器を学習
    labels = target.predict(features)

    # model = LV1UserDefinedClassifierRandomForest()
    model = LV1UserDefinedClassifier1NNRetry()
    model.fit(features, labels)
    print("\nA clone recognizer was trained.")

    # 学習したクローン認識器を可視化し，精度を評価
    evaluator = LV1_Evaluator()
    visualize_save_dir = path_manager.sampling_method_dir(exe_n=exe_n, method_name=method_name)
    create_dir(visualize_save_dir)
    # evaluator.visualize(model, os.path.join(visualize_save_dir, 'visualize.png'))
    print('visualized')
    evaluator.visualize_missing(model=model, target=target,
                                filename=os.path.join(visualize_save_dir, 'visualize_miss.png'), features=features)
    print("\nThe clone recognizer was visualized and saved to {0} .".format(visualize_save_dir))
    accuracy = evaluator.calc_accuracy(target, model)
    print("\naccuracy: {0}".format(accuracy))

    return accuracy


def exe_clone_one():
    n = 1000
    method_name = METHOD_NAME_democracy_demo

    now_str = datetime.now().strftime('%Y%m%d%H%M%S')
    target_path = 'lv1_targets/classifier_03.png'

    save_path_manager = SavePathManager(save_root_dir='output/' + now_str)

    target = LV1TargetClassifier()
    target.load(target_path)
    exe_clone(target=target, exe_n=n, method_name=method_name, path_manager=save_path_manager)


def exe_clone_all(range_arr, target, save_path_manager: SavePathManager, method_name):
    n_list = []
    acc_list = []
    n_list.append(0)
    acc_list.append(0.0)

    for n in range_arr:
        accuracy = exe_clone(target=target, exe_n=n, method_name=method_name, path_manager=save_path_manager)

        n_list.append(n)
        acc_list.append(accuracy)

    return n_list, acc_list


def save_and_show_graph(graph_dir, n_list, acc_list_list):
    print(DIVIDER)
    print('n list')
    print(n_list)
    print(DIVIDER)

    left = np.array(n_list)

    for acc_list, label_text in acc_list_list:
        acc_height = np.array(acc_list)
        plt.plot(left, acc_height, label=label_text)
    plt.xlabel("n samples")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.ylim(0, 1)
    plt.xscale('log', basex=2)
    plt.legend()
    plt.savefig(os.path.join(graph_dir, 'n_accuracy.png'))
    plt.show(block=False)
    plt.close()


def calc_diff_area(start_n, end_n, start_height, end_height):
    rect_area = (end_n - start_n) * start_height
    triangle_area = ((end_n - start_n) * (end_height - start_height)) / 2

    print(rect_area)
    print(triangle_area)

    return rect_area + triangle_area


def calc_area(n_list, acc_list):
    sum_value = 0

    for i in range(len(n_list) - 1):
        sum_value = sum_value + calc_diff_area(start_n=n_list[i],
                                               end_n=n_list[i + 1],
                                               start_height=acc_list[i],
                                               end_height=acc_list[i + 1])

    return sum_value


def create_output(target_path, save_path_manager):
    target = LV1TargetClassifier()
    target.load(target_path)

    range_arr = []
    for i in range(0, 10):
        range_arr.append(2**i)
        #range_arr.append(i)

    print(DIVIDER)
    print('実行間隔')
    print(range_arr)
    print(DIVIDER)

    # pixel_sweeper_n_list, pixel_sweeper_acc_list = exe_clone_all(range_arr=range_arr, target=target,
    #                                                              save_path_manager=save_path_manager,
    #                                                              method_name=METHOD_NAME_SWEEPER_pixel)

    colorless_n_list, colorless_acc_list = exe_clone_all(range_arr=range_arr, target=target,
                                                     save_path_manager=save_path_manager,
                                                     method_name=METHOD_NAME_SWEEPER_COLORLESS)

    demo_n_list, demo_acc_list = exe_clone_all(range_arr=range_arr, target=target,
                                                     save_path_manager=save_path_manager,
                                                     method_name=METHOD_NAME_democracy)

    # colorless_sweeper_n_list, colorless_sweeper_acc_list = exe_clone_all(range_arr=range_arr, target=target,
    #                                                                      save_path_manager=save_path_manager,
    #                                                                      method_name=METHOD_NAME_SWEEPER_COLORLESS)
    n_list = colorless_n_list

    acc_list_list = [(colorless_acc_list, 'colorless_area' + str(calc_area(n_list=n_list,  acc_list=colorless_acc_list))),
                     (demo_acc_list, 'demo_area' + str(calc_area(n_list=n_list, acc_list=demo_acc_list))),
                     ]

    save_and_show_graph(
        graph_dir=save_path_manager.save_root_dir,
        n_list=n_list,
        acc_list_list=acc_list_list
    )




def write_memo(save_path):
    create_dir(save_path)

    print('メモを書く')
    memo = input()

    path_w = os.path.join(save_path, 'memo.txt')

    with open(path_w, mode='w') as f:
        f.write(memo)

# pathの中にある拡張子pngのファイルをリストで返す。
def LV1_user_load_directory(path):
    file_name = os.listdir(path)
    file_path = []
    for i in file_name:
        extension = i.split('.')[-1]
        if extension in 'png':
            file_path.append(path + '/' + i)
    return file_path


def exe_all_images():
    now_str = datetime.now().strftime('%Y%m%d%H%M%S')
    root_path = 'output/' + now_str
    target_names = []

    write_memo(save_path=root_path)

    target_paths = LV1_user_load_directory('lv1_targets')
    target_paths.sort()

    for target_path in target_paths:
        save_path_manager = SavePathManager(save_root_dir=root_path + '/' + target_path[-6:-4])
        create_output(target_path=target_path, save_path_manager=save_path_manager)
        target_names.append(target_path[-4:-6])


if __name__ == '__main__':
    # exe_all_images()
    exe_clone_one()