import numpy as np
import random

np.set_printoptions(suppress=True)
LABEL_SIZE = 10


# 中心を最小の値として中心から遠ざかるほど値が大きくなる2次元配列を返すメソッド
def get_distribution(x_size, y_size):
    size = 3

    def calc(x, y):
        r = x ** 2 + y ** 2
        return int(np.exp(-r) * (size))

    x0 = np.linspace(-2, 2, x_size)
    x1 = np.linspace(-2, 2, y_size)
    arr = np.zeros((len(x0), len(x1)))
    for i0 in range(x_size):
        for i1 in range(y_size):
            arr[i1, i0] = calc(x0[i0], x1[i1])

    ones_arr = np.ones((x_size, y_size))
    zero_points = arr == 0

    arr = np.where(zero_points, ones_arr, arr)

    return arr


class Board:

    def __init__(self, board_size_x: int, board_size_y: int):
        self.board_size_x = board_size_x
        self.board_size_y = board_size_y
        self.positions = np.zeros((LABEL_SIZE, board_size_x, board_size_y))
        self.sampling_points_all = np.full((board_size_x, board_size_y), True)
        self.max_position_x = board_size_x - 1
        self.max_position_y = board_size_y - 1
        self.integrate_positions = np.ones((board_size_x, board_size_y))

    # 小数をpx値にマッピング
    def mapping_x_y(self, feature_x, feature_y):
        x = int(max(0, min(self.max_position_x, np.round((feature_x + 1) * self.max_position_x / 2))))
        y = int(max(0, min(self.max_position_y, np.round((feature_y + 1) * self.max_position_y / 2))))

        return x, y

    # px値を少数にマッピング
    def mapping_feature_x_y(self, x, y):
        feature_x = ((x + 0.5) / self.board_size_x) * 2 - 1
        feature_y = ((y + 0.5) / self.board_size_y) * 2 - 1

        return feature_x, feature_y

    def open_once_feature(self, feature_x, feature_y, color):
        x, y = self.mapping_x_y(feature_x=feature_x, feature_y=feature_y)

        return self.open_once(x=x, y=y, color=color)

    def open_once_colorless_feature(self, feature_x, feature_y, color):
        x, y = self.mapping_x_y(feature_x=feature_x, feature_y=feature_y)

        return self.open_once_colorless(x=x, y=y, color=color)

    # 点を開示
    def open_once(self, x, y, color):
        # 近傍の点のx,yからの距離を算出
        # 中心を最小の値として中心から遠ざかるほど値が大きくなる2次元配列を作る
        distribution_arr = -get_distribution(x_size=self.board_size_x * 2 + 1, y_size=self.board_size_y * 2 + 1)
        trimming_distribution_arr = distribution_arr[self.board_size_x - x:self.board_size_x * 2 - x,
                                    self.board_size_y - y:self.board_size_y * 2 - y]

        self.positions[color] = self.positions[color] + trimming_distribution_arr
        self.sampling_points_all[x, y] = False

    # # 点を開示　サンプリング有効度が高い
    # def open_once_colorless(self, x, y, color):
    #     print('target and clone are not match')
    #     # 中心を最小の値として中心から遠ざかるほど値が大きくなる2次元配列を作る
    #     distribution_arr = get_distribution(x_size=self.board_size_x * 2 + 1, y_size=self.board_size_y * 2 + 1)
    #     trimming_distribution_arr = distribution_arr[self.board_size_x - x:self.board_size_x * 2 - x,
    #                                 self.board_size_y - y:self.board_size_y * 2 - y]
    #
    #     self.positions[color] = np.maximum(self.positions[color], trimming_distribution_arr * 10)
    #
    #     # self.positions[color][x, y] = OPENED
    #     self.sampling_points_all[x, y] = False

    def calc_integrate_positions(self):
        arr = np.zeros((self.board_size_x, self.board_size_y))

        # for x in range(self.board_size_x):
        #     for y in range(self.board_size_y):
        #
        #         numbers = np.zeros(LABEL_SIZE)
        #         for color in range(LABEL_SIZE):
        #             numbers[color] = self.positions[color][x, y]
        #         numbers = numbers[numbers != 0]
        #
        #         # print('numbers')
        #         # print(numbers)
        #         arr[x, y] = np.std(numbers)
        #
        # arr[np.isnan(arr)] = 0

        for l in range(LABEL_SIZE):
            arr = np.minimum(arr, self.positions[l])

        min_value = np.amin(arr) + 1
        min_arr = np.full((self.board_size_x, self.board_size_y), min_value)
        arr = np.where(self.sampling_points_all, arr, min_arr)

        self.integrate_positions = arr

    def get_optimal_solution(self):
        self.calc_integrate_positions()

        print(self.integrate_positions)

        max_value = np.amax(self.integrate_positions)
        x_arr, y_arr = np.where(self.integrate_positions == max_value)

        x_y_arr = list(zip(x_arr, y_arr))
        random.shuffle(x_y_arr)

        index = random.randrange(len(x_y_arr))

        select_x, select_y = x_y_arr[index]

        self.info()

        return self.mapping_feature_x_y(select_x, select_y)

    def info(self):
        print('----------------------')
        print('サンプリング済点')
        print(self.sampling_points_all)
        print('サンプリング効果度分布')
        print(self.integrate_positions)
        print('----------------------')
