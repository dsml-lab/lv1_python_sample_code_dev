import math

import numpy as np

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
