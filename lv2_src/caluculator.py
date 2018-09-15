import matplotlib.pyplot as plt
import numpy as np
from statistics import mean, median, variance, stdev


def calc_diff_area(start_n, end_n, start_height, end_height):
    rect_area = (end_n - start_n) * start_height
    triangle_area = ((end_n - start_n) * (end_height - start_height)) / 2

    print(rect_area)
    print(triangle_area)

    return rect_area + triangle_area


def calc_area(n_list, acc_list):
    sum_value = 0

    for i in range(len(n_list) - 1):
        sum_value = sum_value + calc_diff_area(start_n=n_list[i],
                                               end_n=n_list[i + 1],
                                               start_height=acc_list[i],
                                               end_height=acc_list[i + 1])

    return sum_value


# accuracyの面積の計算結果を画像で保存する。
def area_statistics(save_path, areas, target_names, x_size, y_size, title):
    fig = plt.figure()
    # ax1にclassifier毎のAccuracy面積のグラフ
    ax1 = fig.add_subplot(2, 1, 1)
    # pixelの正規化
    pixel = [round(i / (x_size * y_size), 1) for i in areas]
    plt.plot(np.array(range(len(pixel))), pixel)
    plt.plot(np.array(range(len(pixel))), pixel, 'o')
    plt.title(title)
    plt.xlabel('n_classifier')
    plt.xticks(np.array(range(len(pixel))), target_names)
    plt.ylabel('pixel')
    plt.ylim(0, 1)
    # 平均,中央値,分散,標準偏差の出力
    ax2 = fig.add_subplot(2, 1, 2)
    plt.text(0, 0.6, 'Mean[' + str(round(mean(areas), 2)) + '.pixel,Rate(' + str(
        round((mean(areas) / (x_size * y_size)) * 100, 2)) + '%)]', fontsize=14)
    plt.text(0, 0.4, 'Median[' + str(round(median(areas), 2)) + '.pixel,Rate(' + str(
        round((median(areas) / (x_size * y_size)) * 100, 2)) + '%)]', fontsize=14)
    plt.text(0, 0.2, 'Variance[' + str(round(variance(areas), 2)) + '.pixel]', fontsize=14)
    plt.text(0, 0.0, 'StandardDeviation[' + str(round(stdev(areas), 2)) + '.pixel,Rate(' + str(
        round((stdev(areas) / (x_size * y_size)) * 100, 2)) + '%)]', fontsize=14)
    ax2.axis('off')
    plt.savefig(save_path)
    plt.close()


# accuacyの面積結果を画像で保存する。
def save_area_text(save_path, area, area_size_x, area_size_y):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.text(0, 1.0, str(save_path.split('/')[2] + '_' + save_path.split('/')[3]), fontsize=14)
    plt.text(0, 0.8, 'Image_size[' + str(area_size_x * area_size_y) + '.pixel]', fontsize=14)
    plt.text(0, 0.6, 'Image_size_X[' + str(area_size_x) + '.pixel]_Y[' + str(area_size_y) + '.pixel]', fontsize=14)
    plt.text(0, 0.4, 'pixel_of_AccuracyArea[' + str(area) + '.pixel]', fontsize=14)
    plt.text(0, 0.2, 'Ratio[' + str(round(area / (area_size_x * area_size_y) * 100, 2)) + '%]', fontsize=14)
    ax.axis('off')
    plt.savefig(save_path)
    plt.close()
