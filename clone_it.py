# coding: UTF-8

import sys
from datetime import datetime

import numpy as np
from PIL import Image
from sklearn import neighbors, svm
from labels import COLOR2ID
from evaluation import IMAGE_SIZE
from evaluation import LV1_Evaluator
# 面積を計算するためにグラフを用いるモジュールをimport
import matplotlib.pyplot as plt
from labels import ID2COLOR
import time
import os
from statistics import mean, median, variance, stdev

# ターゲット認識器を表現するクラス
# ターゲット認識器は2次元パターン（512x512の画像）で与えられるものとする
from region import SVMC10gamma10, KNN1, LV1UserDefinedClassifier, KNN3, KNN5, KNN7
from sweeper_sampling import lv1_user_function_sampling_sweeper_colorless, lv1_user_function_sampling_sweeper, \
    LV1UserDefinedClassifierSVM, lv1_user_function_sampling_meshgrid_rectangular


class LV1_TargetClassifier:

    # ターゲット認識器をロード
    #   filename: ターゲット認識器を表現する画像のファイルパス
    def load(self, filename):
        self.img = Image.open(filename)

    # 入力された二次元特徴量に対し，その認識結果（クラスラベルID）を返す
    def predict_once(self, x1, x2):
        h = IMAGE_SIZE // 2
        x = max(0, min(IMAGE_SIZE - 1, np.round(h * x1 + h)))
        y = max(0, min(IMAGE_SIZE - 1, np.round(h - h * x2)))
        return COLOR2ID(self.img.getpixel((x, y)))

    # 入力された二次元特徴量の集合に対し，各々の認識結果を返す
    def predict(self, features):
        labels = np.zeros(features.shape[0])
        for i in range(0, features.shape[0]):
            labels[i] = self.predict_once(features[i][0], features[i][1])
        return np.int32(labels)


# クローン認識器を表現するクラス
# このサンプルコードでは単純な 1-nearest neighbor 認識器とする（sklearnを使用）
# 下記と同型の fit メソッドと predict メソッドが必要
class LV1_UserDefinedClassifier:

    def __init__(self):
        self.clf = neighbors.KNeighborsClassifier(n_neighbors=1)
        # self.clf = svm.SVC()

    # クローン認識器の学習
    #   (features, labels): 訓練データ（特徴量とラベルのペアの集合）
    def fit(self, features, labels):
        self.clf.fit(features, labels)

    # 未知の二次元特徴量を認識
    #   features: 認識対象の二次元特徴量の集合
    def predict(self, features):
        labels = self.clf.predict(features)
        return np.int32(labels)


# ターゲット認識器に入力する二次元特徴量をサンプリングする関数
#   n_samples: サンプリングする特徴量の数
def LV1_user_function_sampling(n_samples=1):
    features = np.zeros((n_samples, 2))
    for i in range(0, n_samples):
        # このサンプルコードでは[-1, 1]の区間をランダムサンプリングするものとする
        features[i][0] = 2 * np.random.rand() - 1
        features[i][1] = 2 * np.random.rand() - 1
    return np.float32(features)


# 面積のグラフ化
def LV1_user_accuracy_plot(accuracy_list, n, path):
    label = [i / 255 for i in ID2COLOR[0]]
    plt.plot(n, accuracy_list, linewidth=1, color=label)
    plt.fill_between(np.array(n), np.array(accuracy_list), facecolor=label)
    plt.title('Accuracy_data')
    plt.xlabel('n_samples')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.xlim(n[0], n[-1])
    # plt.xscale('log')
    plt.savefig(path)
    plt.close()


def LV1_user_plot_cut(path, save_path):
    im = Image.open(path)
    cm = im
    im_crop = im.crop((81, 59, 576, 427))
    # 見える
    cm_crop = cm.crop((80, 58, 577, 428))
    im_crop.save(save_path, quality=100)
    cm_crop.save(save_path.replace('.png', '_line.png'), quality=100)


# cutした画像から指定したpixel数をが数える。
def LV1_user_area_pixel_count(path, save_path):
    # pngがRGBAになるため、jpegのRGBにコンバート
    im = Image.open(path).convert('RGB')
    # 画像サイズを取得して数える。
    size = im.size
    # print('画像サイズ(',size,')_x(',size[0],')_y(',size[1],')')
    X = range(size[0])
    Y = range(size[1])
    wh = Image.open('./none.png')
    count = 0
    for x in X:
        for y in Y:
            # print('xy_norm(',xn,yn,'):xy(',x,y,')')
            pixel = im.getpixel((x, y))
            # print('False:',pixel)
            if pixel == ID2COLOR[0]:
                wh.putpixel((x, y), ID2COLOR[0])
                # print('True:',pixel)
                count += 1
            # print(round(xn,3),round(yn,3),label)
    # 白い画像にplotしてみて確認する。
    wh_re = wh.resize((int(size[0]), int(size[1])))
    wh_re.save(save_path, quality=100)
    return count, size


# accuacyの面積結果を画像で保存する。
def LV1_user_area_count_text(path, pixel_count, area_size):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.text(0, 1.0, str(path.split('/')[2] + '_' + path.split('/')[3]), fontsize=14)
    plt.text(0, 0.8, 'Image_size[' + str(area_size[0] * area_size[1]) + '.pixel]', fontsize=14)
    plt.text(0, 0.6, 'Image_size_X[' + str(area_size[0]) + '.pixel]_Y[' + str(area_size[1]) + '.pixel]', fontsize=14)
    plt.text(0, 0.4, 'pixel_of_AccuracyArea[' + str(pixel_count) + '.pixel]', fontsize=14)
    plt.text(0, 0.2, 'Ratio[' + str(round(pixel_count / (area_size[0] * area_size[1]) * 100, 2)) + '%]', fontsize=14)
    ax.axis('off')
    plt.savefig(path)
    plt.close()


# accuacyの面積の計算結果を画像で保存する。
def LV1_user_area_statistics(path, area_pixel, label, size, title):
    fig = plt.figure()
    # ax1にclassifier毎のAccuracy面積のグラフ
    ax1 = fig.add_subplot(2, 1, 1)
    # pixelの正規化
    pixel = [round(i / (size[0] * size[1]), 1) for i in area_pixel]
    plt.plot(np.array(range(len(pixel))), pixel)
    plt.plot(np.array(range(len(pixel))), pixel, 'o')
    plt.title(title)
    plt.xlabel('n_classifier')
    plt.xticks(np.array(range(len(pixel))), label)
    plt.ylabel('pixel')
    plt.ylim(0, 1)
    # 平均,中央値,分散,標準偏差の出力
    ax2 = fig.add_subplot(2, 1, 2)
    plt.text(0, 0.8, str(path.split('/')[1]), fontsize=14)
    plt.text(0, 0.6, 'Mean[' + str(round(mean(area_pixel), 2)) + '.pixel,Rate(' + str(
        round((mean(area_pixel) / (size[0] * size[1])) * 100, 2)) + '%)]', fontsize=14)
    plt.text(0, 0.4, 'Median[' + str(round(median(area_pixel), 2)) + '.pixel,Rate(' + str(
        round((median(area_pixel) / (size[0] * size[1])) * 100, 2)) + '%)]', fontsize=14)
    plt.text(0, 0.2, 'Variance[' + str(round(variance(area_pixel), 2)) + '.pixel]', fontsize=14)
    plt.text(0, 0.0, 'StandardDeviation[' + str(round(stdev(area_pixel), 2)) + '.pixel,Rate(' + str(
        round((stdev(area_pixel) / (size[0] * size[1])) * 100, 2)) + '%)]', fontsize=14)
    ax2.axis('off')
    plt.savefig(path)
    plt.close()


# pathの下にnameのファイルを作る。
def LV1_user_make_directory(path, name):
    if not os.path.exists(path + '/' + name):
        os.mkdir(path + '/' + name)
    return path + '/' + name


# pathの中にある拡張子pngのファイルをリストで返す。
def LV1_user_load_directory(path):
    file_name = os.listdir(path)
    file_path = []
    for i in file_name:
        extension = i.split('.')[-1]
        if extension in 'png':
            file_path.append(path + '/' + i)
    return file_path


def create_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def write_memo(save_path, method_name):
    create_dir(save_path)

    print('メモを書く')
    memo = input()
    memo = memo + "\n" + method_name

    path_w = os.path.join(save_path, 'memo.txt')

    with open(path_w, mode='w') as f:
        f.write(memo)


def main():
    '''
        if len(sys.argv) < 3:
            print("usage: clone.py /target/classifier/image/path /output/image/path")
            exit(0)
        '''
    # このプログラムファイルの名前と同じdirectoryを作り、その中に結果を保存する。
    now = datetime.now().strftime('%Y%m%d%H%M%S')
    output_path = './output/area_' + now

    create_dir(output_path)

    grid = 'grid'
    colorless = 'colorless'
    sweeper = 'sweeper'

    method_names = [grid, colorless, sweeper]

    for max_value in range(20, 500, 10):
        for method_name in method_names:
            # directory_name = input('作成するdirectoryを入力してください>>>>')
            directory_name = method_name + '_max' + str(max_value) + '_' + now
            directory_path = LV1_user_make_directory(output_path, directory_name)
            # print(directory_path)

            # Lv1_targetsに存在する画像ファイル名を取得する。
            path = './lv1_targets'
            target_image = LV1_user_load_directory(path)
            # print(target_image)

            target_image.sort()
            print(target_image)

            # ターゲット認識器を用意
            target = LV1_TargetClassifier()

            # 面積のpixel数を格納するlist
            area_pixel = []
            last_size = 0
            target_name = []


            # target.load(load_path)をLv1_targetsに含まれる画像毎に指定する。
            for i in target_image:
                # 全部のtargetsをやりたくないときはここをいじって。
                # if i.split('/')[-1].replace('.png','') == 'classifier_03':break
                target_name.append(i.split('/')[-1].split('_')[-1].replace('.png', ''))
                target.load(i)

                # 入力したdirectoryにtarget_image毎のdirectoryを作成する。
                target_directory = LV1_user_make_directory(directory_path, i.split('/')[-1].replace('.png', ''))
                # print(target_directory)

                # 学習したクローン認識器を可視化した画像を保存するためのファイルを作成。
                clone_image_directory = LV1_user_make_directory(target_directory, 'clone_image')
                # print(clone_image_directory)

                # accuracyの結果を保存するフォルダを作成する。
                accuracy_directory = LV1_user_make_directory(target_directory, 'accuracy_area')
                # print(accuracy_directory)

                # ターゲット認識器への入力として用いる二次元特徴量を用意
                # このサンプルコードではひとまず1000サンプルを用意することにする
                # Nごとのaccuracyを格納する配列を用意する。面積を計算するため。
                accuracy_list = []

                # Nにサンプル数の配列を指定する。
                # ↓N=[1,100]は実行時間を短くしたために書いてます。
                # ↓間隔を自分で試したい場合はいじってください。下記の[1~10000]の配列を使う場合はコメントして。
                # N = [1,100]
                # ↓[1,100,200,･･･,10000]までを100間隔でおいた配列がコメントを外すと生成されます。

                # N1 = np.array([1])
                # N2 = np.arange(100, 1001, 100)
                # N = np.hstack((N1, N2))

                # N1 = np.arange(1, 10, 1)
                # N2 = np.arange(10, 100, 10)
                # N3 = np.arange(100, 1001, 100)
                # N = np.hstack((N1, N2))
                # N = np.hstack((N, N3))

                n1 = np.array([1])
                n2 = np.arange(10, max_value + 1, int(max_value / 10))
                N = np.hstack((n1, n2))
                print(N)

                for n in N:
                    start = time.time()

                    if method_name == grid:
                        features = lv1_user_function_sampling_meshgrid_rectangular(n_samples=n)
                    elif method_name == colorless:
                        features = lv1_user_function_sampling_sweeper(n_samples=n, target_model=target, exe_n=n)
                    elif method_name == sweeper:
                        features = lv1_user_function_sampling_sweeper_colorless(n_samples=n, target_model=target, exe_n=n)
                    else:
                        raise ValueError

                    # ターゲット認識器に用意した入力特徴量を入力し，各々に対応するクラスラベルIDを取得
                    labels = target.predict(features)

                    # クローン認識器を学習
                    model = LV1UserDefinedClassifierSVM()
                    model.fit(features, labels)

                    # 学習したクローン認識器を可視化し，精度を評価
                    evaluator = LV1_Evaluator()
                    # 可視化した画像を保存。
                    output_path = clone_image_directory + '/' + directory_name + '_[output_(' + str(n) + ')].png'
                    evaluator.visualize_missing(model=model, filename=output_path, features=features, target=target)

                    # accuracyを配列に格納。
                    accuracy = evaluator.calc_accuracy(target, model)
                    accuracy_list.append(accuracy)

                    end = time.time()
                    print('終了:', i.split('/')[2].replace('.png', ''), '_(', n, ')[', round((end - start), 2), '(sec)]')

                # accuracyの面積グラフを作成して保存
                area_path = accuracy_directory + '/' + directory_name + '_(accuracy_area).png'
                area_features = LV1_user_accuracy_plot(accuracy_list, N, area_path)

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

            statistics_path = directory_path + '/' + directory_name + '_(statistics).png'
            statistics = LV1_user_area_statistics(statistics_path, area_pixel, target_name, last_size, title=method_name)


# クローン処理の実行
# 第一引数でターゲット認識器を表す画像ファイルのパスを，
# 第二引数でクローン認識器の可視化結果を保存する画像ファイルのパスを，
# それぞれ指定するものとする
if __name__ == '__main__':
    main()
