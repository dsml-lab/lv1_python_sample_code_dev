import math

import numpy as np
from tqdm import trange

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


def extract_features_from_images(data_set, extractor, all_image_size, dimension_size):
    # まず，画像データセット中の全画像から特徴量を抽出する
    # 本サンプルコードでは処理時間短縮のため先頭5,000枚のみを対象とする
    # 不要なら行わなくても良い
    all_features = np.zeros((all_image_size, dimension_size))
    all_image_ids = np.zeros(all_image_size)

    for i in trange(0, all_image_size):
        f = data_set.get_feature(i, extractor)
        all_features[i] = f  # 画像番号と特徴量の組を保存
        all_image_ids[i] = i
        print('all features shape')
        print(all_features.shape)

    return all_features, all_image_ids


def convert_list_from_numpy(features, image_ids):
    feature_list = []

    for i in range(len(features)):
        feature_list.append((image_ids[i], features[i]))

    return feature_list


# def convert_numpy_from_list(features_list):
#     features = np.zeros((len(features_list), len(features_list[0])))
#
#     for i in trange(0, len(features_list)):
#         features[i] = features_list[i][1]
#
#     return features


def lv3_user_function_sampling_democracy(data_set, extractor, n_samples, target_model, exe_n, label_table):
    if n_samples <= 0:
        raise ValueError

    elif n_samples <= 1000:
        all_image_size = 5000
        dimension_size = 256

        all_features, all_image_ids = extract_features_from_images(data_set=data_set, extractor=extractor,
                                                                   all_image_size=all_image_size,
                                                                   dimension_size=dimension_size
                                                                   )

        print('n_samples:' + str(n_samples) + ', ' + 'exe_n:' + str(exe_n))

        perm = np.random.permutation(all_image_size)
        new_features = np.zeros((n_samples, dimension_size))
        new_image_ids = np.zeros(n_samples)
        for i in range(0, n_samples):
            new_features[i] = all_features[perm[i]]
            new_image_ids[i] = perm[i]

        target_likelihoods = target_model.predict_proba(convert_list_from_numpy(new_features, new_image_ids))

        if n_samples == exe_n:
            return new_features
        else:
            voters = Parliament.create_lv3_voters(label_table=label_table)
            return new_features, new_image_ids, target_likelihoods, Parliament(
                samplable_features=all_features,
                voter1=voters[0], voter2=voters[1], samplable_feature_ids=all_image_ids)

    elif n_samples > 1:

        old_features, old_image_ids, old_target_likelihoods, parliament = lv3_user_function_sampling_democracy(
            n_samples=n_samples - 1,
            target_model=target_model,
            exe_n=exe_n,
            data_set=data_set,
            extractor=extractor,
            label_table=label_table
            )

        print('n_samples:' + str(n_samples) + ', ' + 'exe_n:' + str(exe_n))

        optimal_feature, optimal_feature_id = parliament.get_optimal_solution(sampled_features=old_features,
                                                          sampled_likelihoods=old_target_likelihoods)

        new_features = optimal_feature
        features = np.vstack((old_features, new_features))

        new_target_likelihoods = target_model.predict_proba(new_features)
        target_likelihoods = np.vstack((old_target_likelihoods, new_target_likelihoods))

        if n_samples == exe_n:
            return features
        else:
            return features, target_likelihoods, parliament
