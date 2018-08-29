# ターゲット認識器に入力する二次元特徴量をサンプリングする関数
#   n_samples: サンプリングする特徴量の数
import numpy as np

from edge_filter import filter_edge
from evaluation import LV1_Evaluator



def lv1_user_function_sampling(n_samples):
    features = np.zeros((n_samples, 2))
    for i in range(0, n_samples):
        # このサンプルコードでは[-1, 1]の区間をランダムサンプリングするものとする
        features[i][0] = 2 * np.random.rand() - 1
        features[i][1] = 2 * np.random.rand() - 1
    return np.float32(features)


def lv1_user_function_sampling_recursion(n_samples):
    print('n_samples:' + str(n_samples))

    new_features = np.zeros((1, 2))
    new_features[0][0] = 2 * np.random.rand() - 1
    new_features[0][1] = 2 * np.random.rand() - 1

    if n_samples == 1:
        return new_features

    old_features = lv1_user_function_sampling_recursion(n_samples=n_samples - 1)

    return np.vstack((old_features, new_features))


# ターゲット認識器に入力する二次元特徴量をサンプリングする関数(格子上)
#   n_samples: サンプリングする特徴量の数
def lv1_user_function_sampling_meshgrid(n_samples):
    features = np.zeros((n_samples, 2))
    # n_samples=10 なら 3
    # n_samples=100 なら 10
    n_samples_sqrt = int(np.sqrt(n_samples))
    n_samples_sq = n_samples_sqrt * n_samples_sqrt
    # 格子ひとつ分の幅
    fragment_size = 2 / (n_samples_sqrt + 1)

    # 格子状に値を入れる
    count = 0
    for j in range(1, n_samples_sqrt + 1):
        for k in range(1, n_samples_sqrt + 1):
            features[count][0] = j * fragment_size - 1
            features[count][1] = k * fragment_size - 1
            count = count + 1

    # 残りはランダムに
    for i in range(n_samples_sq, n_samples):
        features[i][0] = 2 * np.random.rand() - 1
        features[i][1] = 2 * np.random.rand() - 1
    return np.float32(features)



# ターゲット認識器に入力する二次元特徴量をサンプリングする関数(適応的)
#   n_samples: サンプリングする特徴量の数
def lv1_user_function_sampling_and_predict_meshgrid_rectangular_and_edge(n_samples, target, clone_model, grid_n_size, edge_distance):
    # grid_n_size = 500

    if n_samples <= grid_n_size:
        return lv1_user_function_sampling_meshgrid_rectangular(n_samples=n_samples)
    else:
        grid_features = lv1_user_function_sampling_meshgrid_rectangular(n_samples=grid_n_size)

        grid_labels = target.predict(features=grid_features)

        clone_model.fit(grid_features, grid_labels)

        # 学習したクローン認識器を可視化し，精度を評価
        evaluator = LV1_Evaluator()
        clone_img = evaluator.visualize_get_img(clone_model)

        edge_img = filter_edge(img=clone_img)
        edge_features = evaluator.edge_img_to_edge_features(edge_img=edge_img, edge_distance=edge_distance)

        print('edge_features size: ' + str(len(edge_features)))

        print('grid shape' + str(grid_features.shape))
        print('edge shape' + str(edge_features.shape))

        return np.vstack((grid_features, edge_features))


