from scipy.spatial import Delaunay

import numpy as np
import random

'''
点がもつ情報
score
'''

np.set_printoptions(suppress=True)
OPENED = 1000.
COLORLESS = 10
LABEL_SIZE = 11


class Board:

    def __init__(self, board_size):
        self.board_size = board_size
        self.positions = np.zeros((LABEL_SIZE, board_size, board_size))
        self.sampling_points = np.zeros((LABEL_SIZE, board_size, board_size))
        self.sampling_points_all = np.full((board_size, board_size), True)
        self.max_position = board_size - 1
        self.integrate_positions = np.ones((board_size, board_size))

    # 小数をpx値にマッピング
    def mapping_x_y(self, feature_x, feature_y):
        x = int(max(0, min(self.board_size - 1, np.round((feature_x + 1) * (self.board_size - 1) / 2))))
        y = int(max(0, min(self.board_size - 1, np.round((feature_y + 1) * (self.board_size - 1) / 2))))

        print('------------')
        print('board_size: ' + str(self.board_size))
        print('変換前')
        print('feature_x: ' + str(feature_x))
        print('feature_y: ' + str(feature_y))
        print('変換後')
        print('x: ' + str(x))
        print('y: ' + str(y))
        print('------------')

        return x, y

    # px値を少数にマッピング
    def mapping_feature_x_y(self, x, y):
        feature_x = ((x + 0.5) / self.board_size) * 2 - 1
        feature_y = ((y + 0.5) / self.board_size) * 2 - 1

        print('------------')
        print('変換前')
        print('x: ' + str(x))
        print('y: ' + str(y))
        print('変換後')
        print('feature_x: ' + str(feature_x))
        print('feature_y: ' + str(feature_y))
        print('------------')

        return feature_x, feature_y

    def open_once_feature(self, feature_x, feature_y, color=COLORLESS):
        x, y = self.mapping_x_y(feature_x=feature_x, feature_y=feature_y)

        return self.open_once(x=x, y=y, color=color)

    # 点を開示
    def open_once(self, x, y, color=COLORLESS):

        print('------------')
        print('開示')
        print('x: ' + str(x))
        print('y: ' + str(y))
        print('------------')

        # x, yを1次元上の値に直す
        # position = self.board_size * y + x

        # 全ての点のx,yからの距離を計算
        # self.positions[1:3, 1:3] = 1
        for i in range(0, 3):
            self.positions[color][max(x - i, 0):min(x + i + 1, self.board_size),
            max(y - i, 0):min(y + i + 1, self.board_size)] += 1
        self.positions[color][x, y] += OPENED
        self.sampling_points[color][x, y] = OPENED
        self.sampling_points_all[x, y] = False

    def init_open(self):
        # あらかじめ角に点を打つ
        self.open_once(x=0, y=0, color=COLORLESS)
        self.open_once(x=0, y=self.max_position, color=COLORLESS)
        self.open_once(x=self.max_position, y=0, color=COLORLESS)
        self.open_once(x=self.max_position, y=self.max_position, color=COLORLESS)

        # あらかじめ十字に点を打つ
        self.open_once(x=self.max_position // 2, y=0, color=COLORLESS)
        self.open_once(x=0, y=self.max_position // 2, color=COLORLESS)
        self.open_once(x=self.max_position, y=self.max_position // 2, color=COLORLESS)
        self.open_once(x=self.max_position // 2, y=self.max_position, color=COLORLESS)

    def print(self):
        # for c in range(LABEL_SIZE):
        #     print('------------')
        #     print('分布')
        #     print('ラベル' + str(c))
        #     print(self.positions[c])
        #     print('------------')
        #     print('サンプリング点')
        #     print(self.sampling_points[c])
        #     print('------------')

        print('------------')
        print('総合的な分布')
        print(self.integrate_positions)
        print('------------')

    def get_convex_hull(self):
        for points in self.sampling_points:
            x_arr, y_arr = np.where(points == OPENED)
            if len(x_arr) > 4:
                l = [list(a) for a in zip(x_arr, y_arr)]

                print(l)

                hulls = Delaunay(l).convex_hull
                print(hulls)

    def calc_integrate_positions(self):
        arr = np.zeros((self.board_size, self.board_size))

        for i in range(LABEL_SIZE):
            arr = np.absolute(arr - self.positions[i])

        max_value = np.amax(arr) + 1
        max_arr = np.full((self.board_size, self.board_size), max_value)

        self.integrate_positions = np.where(self.sampling_points_all, arr, max_arr)

    def get_optimal_solution(self):
        self.calc_integrate_positions()

        # print(np.amin(integrate_positions))

        min_value = np.amin(self.integrate_positions)
        x_arr, y_arr = np.where(self.integrate_positions == min_value)
        # print(x_arr)
        # print(y_arr)

        index = random.randrange(len(x_arr))

        self.print()

        print('x: ' + str(x_arr[index]))
        print('y: ' + str(y_arr[index]))
        print('選択した値: ' + str(self.integrate_positions[x_arr[index], y_arr[index]]))

        return self.mapping_feature_x_y(x_arr[index], y_arr[index])


def main():
    board_size = 512
    b = Board(board_size=board_size)
    b.print()

    # b.init_open()
    print('初期化')
    b.print()

    # b.init_open()

    for i in range(5):
        b.open_once(i, 3, 2)

    # b.open_once(2, 3, 2)
    # b.print()
    #
    # b.open_once(6, 6, 2)
    # b.print()
    #
    # b.open_once(1, 1, 9)
    # b.print()

    b.calc_integrate_positions()
    b.print()

    b.get_convex_hull()


def check_nan():
    board_size = 10
    b = Board(board_size=board_size)

    b.open_once(0, 3, 2)
    b.open_once(1, 3, 2)
    b.open_once(2, 3, 2)

    b.print()

    b.calc_integrate_positions()

    b.print()





if __name__ == '__main__':
    check_nan()
