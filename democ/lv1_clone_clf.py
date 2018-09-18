import numpy as np
from sklearn import svm, neighbors, tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.neural_network import MLPClassifier

from democ.parliament import Parliament


class LV1UserDefinedClassifierMLP1000HiddenLayerUndiscoveredLabel:
    def __init__(self, label_size):
        self.clf = MLPClassifier(solver="lbfgs", hidden_layer_sizes=1000)
        self.all_labels = range(label_size)

    # クローン認識器の学習
    #   (features, labels): 訓練データ（特徴量とラベルのペアの集合）
    def fit(self, features, labels):
        flat_labels = labels.flatten()
        undiscovered_labels = set(self.all_labels) - set(flat_labels)
        print('undiscovered_labels')
        print(undiscovered_labels)

        voters = Parliament.create_lv1_voters()
        parliament = Parliament(dimension=2,
                                label_size=10,
                                samplable_features=Parliament.get_samplable_features_2_dimension(
                                    image_size=Parliament.get_image_size(len(undiscovered_labels))
                                ),
                                voter1=voters[0],
                                voter2=voters[1]
                                )

        # 未発見ラベルをどこかに割り振る
        for u_discover_label in undiscovered_labels:
            # 次にサンプリングする点としていいと思われる点を取得

            optimal_feature = parliament.get_optimal_solution(
                sampled_features=features,
                sampled_likelihoods=labels
            )
            new_features = np.zeros((1, 2))
            new_features[0][0] = optimal_feature[0]
            new_features[0][1] = optimal_feature[1]
            features = np.vstack((features, new_features))
            labels = np.vstack((labels, u_discover_label))

        print('labels: ')
        print(labels)

        self.clf.fit(features, labels)

        # 未知の二次元特徴量を認識
        #   features: 認識対象の二次元特徴量の集合

    def predict(self, features):
        labels = self.clf.predict(features)
        return np.int32(labels)
