import os

from democ.lv1_clf import LV1UserDefinedClassifierMLP1000HiddenLayer
from democ.sampling import lv1_user_function_sampling_democracy
from lv1_src.caluculator import calc_area, save_area_text, area_statistics
from lv1_src.evaluation_lv1 import LV1Evaluator
from lv1_src.lv1_defined import LV1TargetClassifier
from lv1_src.path_manage import create_dir, load_directories, get_root_dir


def run_clone(target_path, n, visualize_directory):
    # ターゲット認識器を用意
    target = LV1TargetClassifier()
    target.load(target_path)  # ターゲット認識器としてロード

    # ターゲット認識器への入力として用いる二次元特徴量を用意
    # このサンプルコードではひとまず1000サンプルを用意することにする
    features, likelihoods = lv1_user_function_sampling_democracy(n_samples=n, exe_n=n, target_model=target)
    print("\n{0} features were sampled.".format(n))

    print("\nThe sampled features were recognized by the target recognizer.")

    # クローン認識器を学習
    model = LV1UserDefinedClassifierMLP1000HiddenLayer()
    model.fit(features, likelihoods)
    print("\nA clone recognizer was trained.")

    create_dir(visualize_directory)

    # 学習したクローン認識器を可視化し，精度を評価
    evaluator = LV1Evaluator()
    # evaluator.visualize(model, visualize_directory)
    evaluator.visualize_missing(model=model, target=target, filename=os.path.join(visualize_directory, 'missing.png'),
                                features=features)
    print('visualized')
    accuracy = evaluator.calc_accuracy(target, model)
    print("Accuracy: {0}".format(accuracy))

    return accuracy


def run_clone_area(target_path, save_area_path, n_list):
    acc_list = []
    for n in n_list:
        acc = run_clone(n=n, target_path=target_path,
                        visualize_directory=os.path.join(save_area_path, str(n)))
        acc_list.append(acc)

    acc_area = calc_area(n_list=n_list, acc_list=acc_list)
    ratio = acc_area / max(n_list)

    save_area_text(
        save_path=os.path.join(save_area_path, 'area_text.png'),
        area=acc_area,
        area_size_x=max(n_list),
        area_size_y=1
    )

    return acc_area, ratio


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
    run_clone_area_each_targets(targets_path='lv1_targets', save_parent_path=get_root_dir())
