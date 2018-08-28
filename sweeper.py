import numpy as np
import random


np.set_printoptions(suppress=True)
OPENED = 1000.
LABEL_SIZE = 10


def get_distribution(size=10):
    def calc(x, y):
        r = x ** 2 + y ** 2
        return int(np.exp(-r) * 10 ** 2)

    xn = size
    x0 = np.linspace(-2, 2, xn)
    x1 = np.linspace(-2, 2, xn)
    arr = np.zeros((len(x0), len(x1)))
    for i0 in range(xn):
        for i1 in range(xn):
            arr[i1, i0] = calc(x0[i0], x1[i1])

    return arr


class Board:

    def __init__(self, board_size):
        self.board_size = board_size
        self.positions = np.zeros((LABEL_SIZE, board_size, board_size))
        self.sampling_points_all = np.full((board_size, board_size), True)
        self.max_position = board_size - 1
        self.integrate_positions = np.ones((board_size, board_size))

    # 小数をpx値にマッピング
    def mapping_x_y(self, feature_x, feature_y):
        x = int(max(0, min(self.max_position, np.round((feature_x + 1) * self.max_position / 2))))
        y = int(max(0, min(self.max_position, np.round((feature_y + 1) * self.max_position / 2))))

        return x, y

    # px値を少数にマッピング
    def mapping_feature_x_y(self, x, y):
        feature_x = ((x + 0.5) / self.board_size) * 2 - 1
        feature_y = ((y + 0.5) / self.board_size) * 2 - 1

        return feature_x, feature_y

    def open_once_feature(self, feature_x, feature_y, color):
        x, y = self.mapping_x_y(feature_x=feature_x, feature_y=feature_y)

        return self.open_once(x=x, y=y, color=color)

    def open_once_colorless_feature(self, feature_x, feature_y):
        x, y = self.mapping_x_y(feature_x=feature_x, feature_y=feature_y)

        return self.open_once_colorless(x=x, y=y)

    # 点を開示
    def open_once(self, x, y, color):

        # 近傍の点のx,yからの距離を算出
        distribution_arr = get_distribution(size=self.board_size * 2 + 1)  # 中心を最大の値として中心から遠ざかるほど値が小さくなる2次元配列を作る
        trimming_distribution_arr = distribution_arr[self.board_size - x:self.board_size * 2 - x,
                              self.board_size - y:self.board_size * 2 - y]

        self.positions[color] = self.positions[color] + trimming_distribution_arr

        self.positions[color][x, y] += OPENED
        self.sampling_points_all[x, y] = False

    # 点を開示
    def open_once_colorless(self, x, y):
        self.sampling_points_all[x, y] = False

    def print(self):
        print('サンプリング点')
        print(self.sampling_points_all)
        print('------------')
        print('総合的な分布')
        self.calc_integrate_positions()
        print(self.integrate_positions)
        print('------------')

    def calc_integrate_positions(self):
        arr = np.zeros((self.board_size, self.board_size))

        for i in range(LABEL_SIZE):
            arr = np.absolute(arr - self.positions[i])

        max_value = np.amax(arr) + 1
        max_arr = np.full((self.board_size, self.board_size), max_value)

        self.integrate_positions = np.where(self.sampling_points_all, arr, max_arr)

    def adjust(self, value):
        return max(0, min(value, self.board_size))

    def get_optimal_solution(self):
        self.calc_integrate_positions()

        min_value = np.amin(self.integrate_positions)
        x_arr, y_arr = np.where(self.integrate_positions == min_value)

        x_y_arr = list(zip(x_arr, y_arr))
        random.shuffle(x_y_arr)

        index = random.randrange(len(x_y_arr))

        select_x, select_y = x_y_arr[index]

        return self.mapping_feature_x_y(select_x, select_y)
