import os

from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

from clone_it import LV1_user_load_directory, LV1_user_accuracy_plot, LV1_user_plot_cut, LV1_user_area_pixel_count, LV1_user_area_count_text, LV1_user_area_statistics
from evaluation import LV1_Evaluator
from region import lv1_user_function_sampling_region, \
    SavePathManager, LV1UserDefinedClassifier, create_dir, LV1TargetClassifier, DIVIDER, \
    lv1_user_function_sampling_sweeper
from sampling import lv1_user_function_sampling_meshgrid_rectangular, lv1_user_function_sampling

METHOD_NAME_REGION = 'lv1_user_function_sampling_region'
METHOD_NAME_SWEEPER = 'lv1_user_function_sampling_sweeper'
METHOD_NAME_GRID = 'lv1_user_function_sampling_meshgrid_rectangular'
METHOD_NAME_RANDOM = 'lv1_user_function_sampling'

area_pixel = []

def get_features(target, exe_n,
                 method_name, path_manager):

    if method_name == METHOD_NAME_REGION:
        return lv1_user_function_sampling_region(n_samples=exe_n, target_model=target, exe_n=exe_n,
                                                     method_name=method_name, path_manager=path_manager)

    if method_name == METHOD_NAME_SWEEPER:
        features, board = lv1_user_function_sampling_sweeper(n_samples=exe_n, target_model=target, exe_n=exe_n,
                                                     method_name=method_name, path_manager=path_manager)
        return features

    if method_name == METHOD_NAME_GRID:
        return lv1_user_function_sampling_meshgrid_rectangular(n_samples=exe_n)

    if method_name == METHOD_NAME_RANDOM:
        return lv1_user_function_sampling(n_samples=exe_n)


def exe_clone(target, exe_n, method_name, path_manager: SavePathManager):
    # ターゲット認識器への入力として用いる二次元特徴量を用意
    features = get_features(target=target, exe_n=exe_n, method_name=method_name, path_manager=path_manager)

    print(features)
    print(features.shape)
    print(features[0])
    #
    print("\n{0} features were sampled.".format(exe_n))

    # クローン認識器を学習
    labels = target.predict(features)

    model = LV1UserDefinedClassifier()
    model.fit(features, labels)
    print("\nA clone recognizer was trained.")

    # 学習したクローン認識器を可視化し，精度を評価
    evaluator = LV1_Evaluator()
    visualize_save_dir = path_manager.sampling_method_dir(exe_n=exe_n, method_name=method_name)
    create_dir(visualize_save_dir)
    evaluator.visualize(model, os.path.join(visualize_save_dir, 'visualize.png'))
    print('visualized')
    evaluator.visualize_missing(model=model, target=target,
                                filename=os.path.join(visualize_save_dir, 'visualize_miss.png'), features=features)
    print("\nThe clone recognizer was visualized and saved to {0} .".format(visualize_save_dir))
    accuracy = evaluator.calc_accuracy(target, model)
    print("\naccuracy: {0}".format(accuracy))

    return accuracy


def exe_clone_one():
    n = 10
    method_name = METHOD_NAME_SWEEPER

    now_str = datetime.now().strftime('%Y%m%d%H%M%S')
    target_path = 'lv1_targets/classifier_01.png'

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


def save_and_show_graph(graph_dir, n_list, sweeper_acc_list, grid_acc_list, random_acc_list):

    print(DIVIDER)
    print('n list')
    print(n_list)
    print('sweeper_acc_list')
    print(sweeper_acc_list)
    print(DIVIDER)

    left = np.array(n_list)
    sweeper_acc_height = np.array(sweeper_acc_list)
    grid_acc_height = np.array(grid_acc_list)
    random_acc_height = np.array(random_acc_list)
    plt.plot(left, sweeper_acc_height, label='sweeper Accuracy')
    plt.plot(left, grid_acc_height, label='grid Accuracy')
    plt.plot(left, random_acc_height, label='random Accuracy')
    plt.xlabel("n samples")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.ylim(0, 1)
    plt.xscale('log')
    plt.legend()
    plt.savefig(os.path.join(graph_dir, 'n_accuracy.png'))
    plt.show()
    plt.close()


def create_output(target_path, save_path_manager):

    target = LV1TargetClassifier()
    target.load(target_path)

    range_arr = []
    for i in range(1, 11):
        range_arr.append(2**i)

    print(DIVIDER)
    print('実行間隔')
    print(range_arr)
    print(DIVIDER)

    sweeper_n_list, sweeper_acc_list = exe_clone_all(range_arr=range_arr, target=target,
                                                   save_path_manager=save_path_manager, method_name=METHOD_NAME_SWEEPER)

    # region_n_list, region_acc_list = exe_clone_all(range_arr=range_arr, target=target,
    #                                                       save_path_manager=save_path_manager, method_name=METHOD_NAME_REGION)

    grid_n_list, grid_acc_list = exe_clone_all(range_arr=range_arr, target=target,
                                                   save_path_manager=save_path_manager, method_name=METHOD_NAME_GRID)

    random_n_list, random_acc_list = exe_clone_all(range_arr=range_arr, target=target,
                                               save_path_manager=save_path_manager, method_name=METHOD_NAME_RANDOM)

    save_and_show_graph(
        graph_dir=save_path_manager.save_root_dir,
        n_list=grid_n_list,
        sweeper_acc_list=sweeper_acc_list,
        grid_acc_list=grid_acc_list,
        random_acc_list=random_acc_list
    )

    draw_area(accuracy_directory=save_path_manager.save_root_dir,
              n_list=sweeper_n_list,
              accuracy_list=sweeper_acc_list,
              method_name=METHOD_NAME_SWEEPER
              )

    draw_area(accuracy_directory=save_path_manager.save_root_dir,
              n_list=grid_n_list,
              accuracy_list=grid_acc_list,
              method_name=METHOD_NAME_GRID
              )

    return draw_area(accuracy_directory=save_path_manager.save_root_dir,
              n_list=random_n_list,
              accuracy_list=random_acc_list,
              method_name=METHOD_NAME_RANDOM
              )


def draw_area(accuracy_directory, accuracy_list, method_name, n_list):
    # accuracyの面積グラフを作成して保存
    area_path = os.path.join(accuracy_directory, method_name + '_accuracy_area.png')
    area_features = LV1_user_accuracy_plot(accuracy_list, n_list, area_path)

    # 面積のグラフをcutする。
    cut_path = area_path.replace('.png', '_cut.png')
    area_cut = LV1_user_plot_cut(area_path, cut_path)

    # accuracyのぶりつぶされたpixelを数える。
    count_path = area_path.replace('.png', '_count.png')
    pixel_count, area_size = LV1_user_area_pixel_count(cut_path, count_path)
    area_pixel.append(pixel_count)

    # accuracyの面積結果を画像で保存する。
    text_path = area_path.replace('.png', '_text.png')
    area_text = LV1_user_area_count_text(text_path, pixel_count, area_size)
    last_size = area_size

    print('画像サイズ[', area_size, ']_x[', area_size[0], ']_y[', area_size[1], ']')
    print('面積pixel[', pixel_count, ']_割合[', round(pixel_count / (area_size[0] * area_size[1]) * 100, 2), '%]')

    return area_size


def write_memo(save_path):
    create_dir(save_path)

    print('メモを書く')
    memo = input()

    path_w = os.path.join(save_path, 'memo.txt')

    with open(path_w, mode='w') as f:
        f.write(memo)


def exe_all_images():
    now_str = datetime.now().strftime('%Y%m%d%H%M%S')
    root_path = 'output/' + now_str
    target_names = []
    last_size = 0

    write_memo(save_path=root_path)

    target_paths = LV1_user_load_directory('lv1_targets')
    target_paths.sort()

    for target_path in target_paths:
        save_path_manager = SavePathManager(save_root_dir= root_path + '/' + target_path[-6:-4])
        last_size = create_output(target_path=target_path, save_path_manager=save_path_manager)
        target_names.append(target_path[-4:-6])

    statistics_path = root_path + '_(statistics).png'
    statistics = LV1_user_area_statistics(statistics_path, area_pixel, target_names, last_size)


if __name__ == '__main__':
    exe_all_images()
