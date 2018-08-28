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
        # self.sampling_points = np.zeros((LABEL_SIZE, board_size, board_size))
        self.sampling_points_all = np.full((board_size, board_size), True)
        self.max_position = board_size - 1
        self.integrate_positions = np.ones((board_size, board_size))

    # 小数をpx値にマッピング
    def mapping_x_y(self, feature_x, feature_y):
        x = int(max(0, min(self.max_position, np.round((feature_x + 1) * self.max_position / 2))))
        y = int(max(0, min(self.max_position, np.round((feature_y + 1) * self.max_position / 2))))

        # print('------------')
        # print('board_size: ' + str(self.board_size))
        # print('変換前')
        # print('feature_x: ' + str(feature_x))
        # print('feature_y: ' + str(feature_y))
        # print('変換後')
        # print('x: ' + str(x))
        # print('y: ' + str(y))
        # print('------------')

        return x, y

    # px値を少数にマッピング
    def mapping_feature_x_y(self, x, y):
        feature_x = ((x + 0.5) / self.board_size) * 2 - 1
        feature_y = ((y + 0.5) / self.board_size) * 2 - 1

        # print('------------')
        # print('変換前')
        # print('x: ' + str(x))
        # print('y: ' + str(y))
        # print('変換後')
        # print('feature_x: ' + str(feature_x))
        # print('feature_y: ' + str(feature_y))
        # print('------------')

        return feature_x, feature_y

    def open_once_feature(self, feature_x, feature_y, color):
        x, y = self.mapping_x_y(feature_x=feature_x, feature_y=feature_y)

        return self.open_once(x=x, y=y, color=color)

    def open_once_colorless_feature(self, feature_x, feature_y):
        x, y = self.mapping_x_y(feature_x=feature_x, feature_y=feature_y)

        return self.open_once_colorless(x=x, y=y)

    # 点を開示
    def open_once(self, x, y, color):

        print('------------')
        print('開示')
        print('x: ' + str(x))
        print('y: ' + str(y))
        print('------------')

        # 近傍の点のx,yからの距離を計算
        # for i in range(0, self.board_size // 2):
        #     self.positions[color][max(x - i, 0):min(x + i + 1, self.board_size),
        #     max(y - i, 0):min(y + i + 1, self.board_size)] += 1

        distribution_arr = get_distribution(size=self.board_size * 2 + 1)  # 正規分布の2次元配列

        trimming_distribution_arr = distribution_arr[self.board_size - x:self.board_size * 2 - x,
                              self.board_size - y:self.board_size * 2 - y]

        print('color')
        print(self.positions[color])

        self.positions[color] = self.positions[color] + trimming_distribution_arr

        # print('------------------')
        # print('トリミング')
        # print(trimming_distribution_arr)
        # print('色ごとの分布')
        # print(self.positions[color])
        # print('------------------')

        self.positions[color][x, y] += OPENED
        # self.sampling_points[color][x, y] = OPENED
        self.sampling_points_all[x, y] = False

    # 点を開示
    def open_once_colorless(self, x, y):
        self.sampling_points_all[x, y] = False

    def print(self):
        # for c in range(LABEL_SIZE):
        #     print('------------')
        #     print('分布')
        #     print('ラベル' + str(c))
        #     print(self.positions[c])
        #     print('------------')

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

        # if self.board_size > 16:
        #     # 周囲の要素の平均が最小の点
        #     min_around_ave = np.amax(self.integrate_positions)
        #     for i, (x, y) in enumerate(x_y_arr):
        #         around_ave = np.average(self.integrate_positions[self.adjust(x - 1):self.adjust(x + 1),
        #                                 self.adjust(y - 1):self.adjust(y + 1)])
        #         if min_around_ave > around_ave:
        #             min_around_ave = around_ave
        #             index = i

        select_x, select_y = x_y_arr[index]

        print('x: ' + str(select_x))
        print('y: ' + str(select_y))
        print('選択した値: ' + str(self.integrate_positions[select_x, select_y]))

        return self.mapping_feature_x_y(select_x, select_y)


def main():
    board_size = 8
    b = Board(board_size=board_size)
    b.print()

    b.open_once_colorless(0, 0)
    b.open_once_colorless(2, 7)

    b.open_once(3, 3, 4)

    b.print()

    # b.open_once(2, 3, 2)
    # b.print()
    #
    # b.open_once(6, 6, 2)
    # b.print()
    #
    # b.open_once(1, 1, 2)
    # b.print()
    #
    # b.open_once(0, 0, 1)
    # b.print()
    #
    # b.open_once(0, 2, 3)
    # b.print()

    b.calc_integrate_positions()
    b.print()


def check_nan():
    board_size = 4
    b = Board(board_size=board_size)

    b.open_once(0, 3, 2)
    b.open_once(1, 3, 2)
    b.open_once(2, 3, 2)

    b.print()

    b.calc_integrate_positions()

    b.print()


if __name__ == '__main__':
    main()
