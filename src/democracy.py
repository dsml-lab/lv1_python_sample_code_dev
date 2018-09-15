import math

import numpy as np
from sklearn.preprocessing import OneHotEncoder

from democ.distance import find_furthest_place
from democ.lv1_clf import LV1UserDefinedClassifierSVM10C10Gamma, LV1UserDefinedClassifier1NN, \
    LV1UserDefinedClassifierMLP1000HiddenLayer

LEVEL_1 = 1
LEVEL_2 = 2
LEVEL_3 = 3


class Voter:
    """投票者クラス"""

    def __init__(self, model, label_size):
        self.model = model
        self.samplable_labels = None
        self.label_size = label_size

    # クローン認識器の学習
    #   (features, labels): 訓練データ（特徴量とラベルのペアの集合）
    def sampled_fit(self, sampled_features, sampled_labels):
        self.model.fit(sampled_features, sampled_labels)  # 学習

    # 未知の二次元特徴量を認識
    #   features: 認識対象の二次元特徴量の集合
    def samplable_predict(self, samplable_features):
        self.samplable_labels = self.__to_one_hot(self.model.predict(samplable_features))  # 予測結果を保持

    def __to_one_hot(self, labels):
        encoder = OneHotEncoder(self.label_size)
        return encoder.fit_transform(np.reshape(labels, (-1, 1))).toarray()


class Parliament:
    """議会クラス"""

    def __get_samplable_features_2_dimension(self, image_size):
        h = image_size // 2
        point_count = image_size * image_size
        samplable_features = np.zeros((point_count, self.dimension))
        for i in range(0, point_count):
            x = i % image_size
            y = i // image_size
            samplable_features[i][0] = np.float32((x - h) / h)
            samplable_features[i][1] = np.float32(-(y - h) / h)
        return np.float32(samplable_features)

    def __init_voters_lv1(self):
        # self.voters.append(Lv1Voter(model=LV1UserDefinedClassifierSVM10C10Gamma()))
        # self.voters.append(Lv1Voter(model=LV1UserDefinedClassifier1NN()))
        # self.voters.append(Voter(model=LV1UserDefinedClassifier7NN()))
        self.voters.append(Voter(model=LV1UserDefinedClassifierMLP1000HiddenLayer(), label_size=10))
        self.voters.append(Voter(model=LV1UserDefinedClassifierMLP1000HiddenLayer(), label_size=10))
        # self.voters.append(Voter(model=LV1UserDefinedClassifierTree1000MaxDepth()))
        # self.voters.append(Voter(model=LV1UserDefinedClassifierRandomForest()))

    def __init__(self, level, exe_n):
        self.voters = []

        if level == LEVEL_1:
            self.dimension = 2
            self.__init_voters_lv1()
            image_size = self.__get_image_size(exe_n=exe_n)
            self.samplable_features = self.__get_samplable_features_2_dimension(image_size=image_size)
        elif level == LEVEL_2:
            self.dimension = 2
            self.__init_voters()
            image_size = self.__get_image_size(exe_n=exe_n)
            self.samplable_features = self.__get_samplable_features_2_dimension(image_size=image_size)

    def get_optimal_solution(self, sampled_features, sampled_labels):
        self.__fit_to_voters(sampled_features=sampled_features, sampled_labels=sampled_labels)
        self.__predict_to_voters()

        label_count_arr = np.zeros((len(self.samplable_features), LABEL_SIZE))  # one hot 形式

        for voter in self.voters:
            samplable_labels = voter.samplable_labels

            label_count_arr = label_count_arr + samplable_labels

        label_count_arr[label_count_arr > 0] = 1

        label_count_arr = label_count_arr.sum(axis=1) # 1次元行列
        label_count_arr[label_count_arr > 1] = 2

        max_value = np.amax(label_count_arr)
        index_list = np.where(label_count_arr == max_value)[0]
        filtered_samplable_features = self.samplable_features[index_list]
        opt_feature = find_furthest_place(sampled_features=sampled_features, filtered_samplable_features=filtered_samplable_features)

        self.delete_samplable_features(delete_feature=opt_feature)

        return opt_feature

    def delete_samplable_features(self, delete_feature):

        index_list = np.where(delete_feature == self.samplable_features)[0]

        # サンプリング候補から除外
        self.samplable_features = np.delete(self.samplable_features, index_list[0], axis=0)

    def __fit_to_voters(self, sampled_features, sampled_labels):
        for i in range(len(self.voters)):
            self.voters[i].sampled_fit(sampled_features=sampled_features, sampled_labels=sampled_labels)

    def __predict_to_voters(self):
        for i in range(len(self.voters)):
            self.voters[i].samplable_predict(samplable_features=self.samplable_features)

    @staticmethod
    def __get_image_size(exe_n):
        return math.ceil(math.sqrt(exe_n)) + 128

        # if exe_n < 128**2:
        #     return 128
        # elif exe_n < 256**2:
        #     return 256
        # else:
        #     return 512


def lv1_user_function_sampling_democracy(n_samples, target_model, exe_n):
    if n_samples < 0:
        raise ValueError

    elif n_samples == 0:
        return np.zeros((0, 2))

    elif n_samples == 1:

        print('n_samples:' + str(n_samples) + ', ' + 'exe_n:' + str(exe_n))
        new_features = np.zeros((1, 2))
        new_features[0][0] = 2 * np.random.rand() - 1
        new_features[0][1] = 2 * np.random.rand() - 1

        if n_samples == exe_n:
            return np.float32(new_features)
        else:
            return np.float32(new_features), Parliament(image_size=int(get_image_size(exe_n)))

    elif n_samples > 1:

        old_features, parliament = lv1_user_function_sampling_democracy(n_samples=n_samples - 1,
                                                                        target_model=target_model,
                                                                        exe_n=exe_n)

        print('n_samples:' + str(n_samples) + ', ' + 'exe_n:' + str(exe_n))

        # target識別器からtargetのラベルを取得
        target_labels = target_model.predict(old_features)

        optimal_feature = parliament.get_optimal_solution(sampled_features=old_features, sampled_labels=target_labels)

        new_features = np.zeros((1, 2))
        new_features[0][0] = optimal_feature[0]
        new_features[0][1] = optimal_feature[1]

        features = np.vstack((old_features, new_features))

        if n_samples == exe_n:
            return np.float32(features)
        else:
            return np.float32(features), parliament


def lv1_user_function_sampling_grid(n_samples):
    features = np.zeros((n_samples, 2))

    x_samples = 0
    y_samples = 0

    # 格子点の個数がもっとも多くなる
    # y_sizeとy_sizeの差がなるべく小さくなる

    for i in range(2, n_samples):
        for j in range(2, n_samples):
            if n_samples >= i * j > x_samples * y_samples and abs(i - j) < 2:  # 格子の縦横の差が2より小さい
                x_samples = i
                y_samples = j

    # 格子ひとつ分の幅
    x_size = 2 / (x_samples + 1)
    y_size = 2 / (y_samples + 1)

    # 格子状に値を入れる
    count = 0
    for j in range(1, x_samples + 1):
        for k in range(1, y_samples + 1):
            features[count][0] = j * x_size - 1
            features[count][1] = k * y_size - 1
            count = count + 1

    # 残りはランダムに
    for i in range(x_samples * y_samples, n_samples):
        features[i][0] = 2 * np.random.rand() - 1
        features[i][1] = 2 * np.random.rand() - 1
    return np.float32(features)


def lv1_user_function_sampling_democracy_and_grid(n_samples, target_model):

    grid_samples = 128

    if grid_samples < n_samples:
        grid_features = lv1_user_function_sampling_grid(grid_samples)

        demo_exe_n = n_samples - grid_samples
        demo_features = lv1_user_function_sampling_democracy(n_samples=demo_exe_n, exe_n=demo_exe_n, target_model=target_model)

        return np.float32(np.vstack((grid_features, demo_features)))
    else:
        return lv1_user_function_sampling_democracy(n_samples=n_samples, exe_n=n_samples, target_model=target_model)
