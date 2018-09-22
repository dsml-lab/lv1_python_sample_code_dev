import math

import numpy as np
from tqdm import trange, tqdm

from democ.parliament import Parliament


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

        target_labels = target_model.predict(new_features)

        if n_samples == exe_n:
            return np.float32(new_features), target_labels
        else:
            voters = Parliament.create_lv1_voters()
            return np.float32(new_features), target_labels, Parliament(
                dimension=2,
                label_size=10,
                samplable_features=Parliament.get_samplable_features_2_dimension(
                    image_size=Parliament.get_image_size(exe_n=exe_n)),
                voter1=voters[0], voter2=voters[1]),

    elif n_samples > 1:

        old_features, old_target_labels, parliament = lv1_user_function_sampling_democracy(n_samples=n_samples - 1,
                                                                                           target_model=target_model,
                                                                                           exe_n=exe_n)

        print('n_samples:' + str(n_samples) + ', ' + 'exe_n:' + str(exe_n))

        optimal_feature = parliament.get_optimal_solution(sampled_features=old_features,
                                                          sampled_likelihoods=old_target_labels)

        new_features = np.zeros((1, 2))
        new_features[0][0] = optimal_feature[0]
        new_features[0][1] = optimal_feature[1]
        features = np.vstack((old_features, new_features))

        new_target_labels = target_model.predict(new_features)
        target_labels = np.vstack((old_target_labels, new_target_labels))

        if n_samples == exe_n:
            return np.float32(features), target_labels
        else:
            return np.float32(features), target_labels, parliament


def lv2_user_function_sampling_democracy(n_samples, target_model, exe_n):
    if n_samples < 0:
        raise ValueError

    elif n_samples == 0:
        return np.zeros((0, 2))

    elif n_samples == 1:

        print('n_samples:' + str(n_samples) + ', ' + 'exe_n:' + str(exe_n))
        new_features = np.zeros((1, 2))
        new_features[0][0] = 2 * np.random.rand() - 1
        new_features[0][1] = 2 * np.random.rand() - 1

        target_likelihoods = target_model.predict_proba(new_features)

        if n_samples == exe_n:
            return np.float32(new_features), target_likelihoods
        else:
            vtrs = Parliament.create_lv2_voters()
            return np.float32(new_features), target_likelihoods, Parliament(
                dimension=2,
                label_size=8,
                samplable_features=Parliament.get_samplable_features_2_dimension(
                    image_size=Parliament.get_image_size(exe_n=exe_n)),
                voter1=vtrs[0], voter2=vtrs[1]),

    elif n_samples > 1:

        old_features, old_target_likelihoods, parliament = lv2_user_function_sampling_democracy(n_samples=n_samples - 1,
                                                                                                target_model=target_model,
                                                                                                exe_n=exe_n)

        print('n_samples:' + str(n_samples) + ', ' + 'exe_n:' + str(exe_n))

        optimal_feature = parliament.get_optimal_solution(sampled_features=old_features,
                                                          sampled_likelihoods=old_target_likelihoods)

        new_features = np.zeros((1, 2))
        new_features[0][0] = optimal_feature[0]
        new_features[0][1] = optimal_feature[1]
        features = np.vstack((old_features, new_features))

        new_target_likelihoods = target_model.predict_proba(new_features)
        target_likelihoods = np.vstack((old_target_likelihoods, new_target_likelihoods))

        if n_samples == exe_n:
            return np.float32(features), target_likelihoods
        else:
            return np.float32(features), target_likelihoods, parliament


def extract_features_from_images(data_set, extractor, all_image_count):
    # まず，画像データセット中の全画像から特徴量を抽出する
    # 本サンプルコードでは処理時間短縮のため先頭all_image_count枚のみを対象とする
    all_features = []

    for i in trange(0, all_image_count):
        f = data_set.get_feature(i, extractor)
        all_features.append((i, f))  # 画像番号と特徴量の組を保存

    return all_features


def convert_list_from_numpy(features, image_ids):
    feature_list = []

    for i in range(len(features)):
        f = features[i].reshape(-1, 1)
        feature_list.append((np.int32(image_ids[i]), f))

    return feature_list


INITIAL_VALUE = 1000


def lv3_user_function_sampling_democracy(data_set, extractor, n_samples, target_model, exe_n, n_labels):
    if n_samples <= 0:
        raise ValueError

    elif n_samples <= INITIAL_VALUE:
        all_image_count = 5000

        all_features = extract_features_from_images(data_set=data_set, extractor=extractor,
                                                    all_image_count=all_image_count
                                                    )

        print('n_samples:' + str(n_samples) + ', ' + 'exe_n:' + str(exe_n))

        perm = np.random.permutation(all_image_count)
        new_features = []
        for i in range(0, n_samples):
            new_features.append(all_features[perm[i]])

        target_likelihoods = target_model.predict_proba(new_features)

        if n_samples == exe_n:
            return new_features
        else:
            voters = Parliament.create_lv3_voters(n_labels=n_labels)
            parliament = Parliament(
                samplable_features=all_features,
                voter1=voters[0], voter2=voters[1])
            for fe in tqdm(new_features):
                parliament.delete_samplable_features(delete_feature=fe)
            return new_features, target_likelihoods, parliament

    elif n_samples > INITIAL_VALUE:
        increase_width = 100

        old_features, old_target_likelihoods, parliament = lv3_user_function_sampling_democracy(
            n_samples=n_samples - increase_width,
            target_model=target_model,
            exe_n=exe_n,
            data_set=data_set,
            extractor=extractor,
            n_labels=n_labels
        )

        print('n_samples:' + str(n_samples) + ', ' + 'exe_n:' + str(exe_n))

        parliament.fit_to_voters(sampled_features=old_features, sampled_likelihoods=old_target_likelihoods)

        optimal_features = []

        for _ in range(increase_width):
            opt_feature = parliament.get_optimal_solution(sampled_features=old_features, n_labels=n_labels)
            old_features.append(opt_feature)
            optimal_features.append(opt_feature)

        new_target_likelihoods = target_model.predict_proba(optimal_features)
        target_likelihoods = np.vstack((old_target_likelihoods, new_target_likelihoods))

        if n_samples == exe_n:
            return old_features
        else:
            return old_features, target_likelihoods, parliament
