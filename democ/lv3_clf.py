import numpy as np

# クローン認識器を表現するクラス
# このサンプルコードでは各クラスラベルごとに単純な 5-nearest neighbor を行うものとする（sklearnを使用）
# 下記と同型の fit メソッドと predict_proba メソッドが必要
from keras import Input, Model, Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.optimizers import adam, SGD
from keras.utils import plot_model
from keras_applications.vgg16 import VGG16
from keras_applications.vgg19 import VGG19
from skimage.feature import local_binary_pattern
from sklearn import neighbors, svm
from sklearn.neural_network import MLPClassifier
from tqdm import trange


class LV3UserDefinedClassifier:

    def __init__(self, n_labels):
        self.n_labels = n_labels
        self.clfs = []
        for i in trange(0, self.n_labels):
            clf = neighbors.KNeighborsClassifier(n_neighbors=5)
            self.clfs.append(clf)

    def __mold_features(self, features):
        temp = []
        for i in trange(0, len(features)):
            temp.append(features[i][1])
        return np.asarray(temp, dtype=np.float32)

    # クローン認識器の学習
    #   (features, likelihoods): 訓練データ（特徴量と尤度ベクトルのペアの集合）
    def fit(self, features, likelihoods):
        features = self.__mold_features(features)
        labels = np.int32(likelihoods >= 0.5)  # 尤度0.5以上のラベルのみがターゲット認識器の認識結果であると解釈する
        for i in range(0, self.n_labels):
            l = labels[:, i]
            self.clfs[i].fit(features, l)

    # 未知の特徴量を認識
    #   features: 認識対象の特徴量の集合
    def predict_proba(self, features):
        features = self.__mold_features(features)
        likelihoods = np.c_[np.zeros(features.shape[0])]
        for i in range(0, self.n_labels):
            p = self.clfs[i].predict_proba(features)
            likelihoods = np.hstack([likelihoods, np.c_[p[:, 1]]])
        likelihoods = likelihoods[:, 1:]
        return np.float32(likelihoods)


class LV3UserDefinedClassifierKNN3:

    def __init__(self, n_labels):
        self.n_labels = n_labels
        self.clfs = []
        for i in trange(0, self.n_labels):
            clf = neighbors.KNeighborsClassifier(n_neighbors=3)
            self.clfs.append(clf)

    def __mold_features(self, features):
        temp = []
        for i in trange(0, len(features)):
            temp.append(features[i][1])
        return np.asarray(temp, dtype=np.float32)

    # クローン認識器の学習
    #   (features, likelihoods): 訓練データ（特徴量と尤度ベクトルのペアの集合）
    def fit(self, features, likelihoods):
        features = self.__mold_features(features)
        labels = np.int32(likelihoods >= 0.5)  # 尤度0.5以上のラベルのみがターゲット認識器の認識結果であると解釈する
        for i in range(0, self.n_labels):
            l = labels[:, i]
            self.clfs[i].fit(features, l)

    # 未知の特徴量を認識
    #   features: 認識対象の特徴量の集合
    def predict_proba(self, features):
        features = self.__mold_features(features)
        likelihoods = np.c_[np.zeros(features.shape[0])]
        for i in range(0, self.n_labels):
            p = self.clfs[i].predict_proba(features)
            likelihoods = np.hstack([likelihoods, np.c_[p[:, 1]]])
        likelihoods = likelihoods[:, 1:]
        return np.float32(likelihoods)


# class LV3UserDefinedClassifierVGG16:
#
#     @staticmethod
#     def build_model(n_labels):
#         # input_tensor = Input(shape=(48, 48, 1))
#         vgg16_model = VGG16(weights='imagenet', include_top=False, input_shape=(48, 48, 3))
#         vgg16_model.summary()
#
#         top_model = Sequential()
#         top_model.add(Flatten(input_shape=vgg16_model.output_shape[1:]))
#         top_model.add(Dense(16, activation='relu'))
#         top_model.add(Dense(8, activation='relu'))
#         top_model.add(Dense(8, activation='relu'))
#         top_model.add(Dense(n_labels, activation='sigmoid'))
#
#         model = Model(input=vgg16_model.input, output=top_model(vgg16_model.output))
#         model.compile(optimizer='rmsprop', loss='binary_crossentropy')
#         model.summary()
#
#         return model
#
#     def __init__(self, n_labels):
#         self.n_labels = n_labels
#         self.clf = self.build_model(n_labels)
#
#     def __mold_features(self, features):
#         temp = []
#         for i in trange(0, len(features)):
#             temp.append(np.reshape(features[i][1], (48, 48, 3)))
#         return np.asarray(temp, dtype=np.float32)
#
#     # クローン認識器の学習
#     #   (features, likelihoods): 訓練データ（特徴量と尤度ベクトルのペアの集合）
#     def fit(self, features, likelihoods):
#         batch_size = 20
#         features = self.__mold_features(features)
#         labels = np.int32(likelihoods >= 0.5)  # 尤度0.5以上のラベルのみがターゲット認識器の認識結果であると解釈する
#         es_cb = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, mode='auto')
#         self.clf.fit(features, labels,
#                      batch_size=batch_size,
#                      epochs=10,
#                      verbose=1,
#                      callbacks = [es_cb]
#                      )
#
#     # 未知の特徴量を認識
#     #   features: 認識対象の特徴量の集合
#     def predict_proba(self, features):
#         features = self.__mold_features(features)
#         likelihoods = self.clf.predict(features, verbose=1)
#         return np.float32(likelihoods)


vgg_input_value = 224
vgg_input_shape = (224, 224, 3)


class VGG16KerasModel:

    @staticmethod
    def build_model(n_labels):
        # input_tensor = Input(shape=(48, 48, 1))
        vgg16_model = VGG16(weights=None, include_top=False,
                           input_tensor=Input(shape=vgg_input_shape))

        top_model = Sequential()
        top_model.add(Flatten(input_shape=vgg16_model.output_shape[1:]))
        # top_model.add(Dense(32, activation='relu'))
        top_model.add(Dense(8, activation='relu'))
        # top_model.add(Dense(8, activation='relu'))
        top_model.add(Dense(n_labels, activation='sigmoid'))

        model = Model(input=vgg16_model.input, output=top_model(vgg16_model.output))
        model.compile(optimizer='rmsprop', loss='binary_crossentropy')
        model.summary()

        return model

    @staticmethod
    def build_model_func(n_labels):
        base_model = VGG16(weights='imagenet', include_top=False,
                           input_tensor=Input(shape=vgg_input_shape))

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        prediction = Dense(n_labels, activation='sigmoid')(x)
        model = Model(inputs=base_model.input, outputs=prediction)

        # fix weights before VGG16 14layers
        for layer in base_model.layers[:15]:
            layer.trainable = False

        model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        # model.summary()

        return model

    def __init__(self, mask):
        self.mask = mask
        self.clf = self.build_model(n_labels=np.sum(mask))

    @staticmethod
    def __mold_features(features):
        temp = []
        for i in trange(0, len(features)):
            temp.append(np.reshape(features[i][1], vgg_input_shape))
        return np.asarray(temp, dtype=np.float32)

    # クローン認識器の学習
    #   (features, likelihoods): 訓練データ（特徴量と尤度ベクトルのペアの集合）
    def fit(self, features, labels):

        batch_size = 20
        features = self.__mold_features(features)

        print('features')
        print(features)
        print(features.shape)

        print('labels')
        print(labels)
        print(labels.shape)

        es_cb = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, mode='auto')
        self.clf.fit(features, labels,
                     batch_size=batch_size,
                     epochs=10,
                     verbose=1,
                     callbacks = [es_cb]
                     )

    # 未知の特徴量を認識
    #   features: 認識対象の特徴量の集合
    def predict_proba(self, features):
        features = self.__mold_features(features)
        likelihoods = self.clf.predict(features, verbose=1)
        return np.float32(likelihoods)


class DenseModel:

    @staticmethod
    def build_model(n_labels):
        # input_tensor = Input(shape=(48, 48, 1))
        # vgg16_model = VGG16(weights=None, include_top=False,
        #                    input_tensor=Input(shape=vgg_input_shape))

        top_model = Sequential()
        top_model.add(Dense(256, activation='relu', input_shape=(256,)))
        top_model.add(Dense(1024, activation='relu'))
        top_model.add(Dense(256, activation='relu'))
        top_model.add(Dense(n_labels, activation='sigmoid'))

        top_model.compile(optimizer='rmsprop', loss='binary_crossentropy')
        top_model.summary()

        return top_model

    def __init__(self, mask):
        self.mask = mask
        self.clf = self.build_model(n_labels=np.sum(mask))

    def __mold_features(self, features):
        temp = []
        for i in trange(0, len(features)):
            temp.append(features[i][1])
        return np.asarray(temp, dtype=np.float32)

    # クローン認識器の学習
    #   (features, likelihoods): 訓練データ（特徴量と尤度ベクトルのペアの集合）
    def fit(self, features, labels):

        batch_size = 20
        features = self.__mold_features(features)

        print('features')
        print(features)
        print(features.shape)

        print('labels')
        print(labels)
        print(labels.shape)

        es_cb = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, mode='auto')
        self.clf.fit(features, labels,
                     batch_size=batch_size,
                     epochs=10,
                     verbose=1,
                     callbacks = [es_cb])

    # 未知の特徴量を認識
    #   features: 認識対象の特徴量の集合
    def predict_proba(self, features):
        features = self.__mold_features(features)
        likelihoods = self.clf.predict(features, verbose=1)
        return np.float32(likelihoods)


class LV3UserDefinedClassifierDivide:

    def __init__(self, labels_all):
        self.divide_label_num = len(labels_all)
        self.labels_all = np.array(labels_all)
        self.clfs = []

        for i in range(len(self.labels_all) // self.divide_label_num):
            # マスクする
            fragment_labels = np.zeros(self.labels_all.shape)
            fragment_labels[i*self.divide_label_num:(i+1)*self.divide_label_num] = 1
            fragment_labels = fragment_labels == 1

            clf = VGG16KerasModel(mask=fragment_labels)
            # clf = DenseModel(mask=fragment_labels)
            self.clfs.append(clf)


    # @staticmethod
    # def __mold_features(features):
    #     temp = []
    #     for i in trange(0, len(features)):
    #         temp.append(features[i][1])
    #     return np.asarray(temp, dtype=np.float32)

    # クローン認識器の学習
    #   (features, likelihoods): 訓練データ（特徴量と尤度ベクトルのペアの集合）
    def fit(self, features, likelihoods):
        labels = np.int32(likelihoods >= 0.5)  # 尤度0.5以上のラベルのみがターゲット認識器の認識結果であると解釈する

        print(labels.shape)

        for i in range(len(self.labels_all) // self.divide_label_num):
            masked_labels = []
            for j in range(len(labels)):
                masked_labels.append(labels[j][self.clfs[i].mask])

            masked_labels = np.array(masked_labels)

            print(masked_labels.shape)
            print(masked_labels)

            self.clfs[i].fit(features=features, labels=masked_labels)

    # 未知の特徴量を認識
    #   features: 認識対象の特徴量の集合
    def predict_proba(self, features):
        # # features = self.__mold_features(features)
        # likelihoods = np.c_[np.zeros(len(features))]
        # for i in range(len(self.labels_all) // self.divide_label_num):
        #     p = self.clfs[i].predict_proba(features)
        #     likelihoods = np.hstack([likelihoods, np.c_[p[:, 1]]])
        # likelihoods = likelihoods[:, 1:]
        likelihoods = np.zeros((len(features), self.labels_all.shape[0]))
        print(likelihoods.shape)

        divide_likelihoods_list = []
        for i in range(len(self.labels_all) // self.divide_label_num):
            divide_likelihoods = self.clfs[i].predict_proba(features=features)
            divide_likelihoods_list.append(divide_likelihoods)

        for i in range(len(likelihoods)):
            line_likelihoods = []
            for divide_likelihoods in divide_likelihoods_list:
                line_likelihoods = line_likelihoods + list(divide_likelihoods[i])

            line_likelihoods = np.array(line_likelihoods)

            likelihoods[i] = line_likelihoods

        return np.float32(likelihoods)


if __name__ == '__main__':
    model = LV3UserDefinedClassifierDivide(labels=np.zeros(248))
