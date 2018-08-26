from scipy.spatial import Delaunay

import numpy as np
import random

'''
点がもつ情報
score
'''

np.set_printoptions(suppress=True)
OPENED = 1.
COLORLESS = 10
LABEL_SIZE = 11


class Board:

    def __init__(self, board_size):
        self.board_size = board_size
        self.positions = np.zeros((LABEL_SIZE, board_size, board_size))
        self.sampling_points = np.zeros((LABEL_SIZE, board_size, board_size))
        self.max_position = board_size - 1
        self.integrate_positions = np.ones((board_size, board_size))

    # 小数をpx値にマッピング
    def mapping_x_y(self, feature_x, feature_y):
        h = self.board_size // 2
        x = int(max(0, min(self.board_size - 1, np.round(h * feature_x + h))))
        y = int(max(0, min(self.board_size - 1, np.round(h - h * feature_y))))

        return x, y

    # px値を少数にマッピング
    def mapping_feature_x_y(self, x, y):
        feature_x = ((x + 0.5) / self.board_size)*2 - 1
        feature_y = ((y + 0.5) / self.board_size)*2 - 1

        return feature_x, feature_y

    def open_once_feature(self, feature_x, feature_y, color=COLORLESS):
        x, y = self.mapping_x_y(feature_x=feature_x, feature_y=feature_y)

        return self.open_once(x=x, y=y, color=color)

    # 点を開示
    def open_once(self, x, y, color=COLORLESS):

        # x, yを1次元上の値に直す
        # position = self.board_size * y + x

        # 全ての点のx,yからの距離を計算
        # self.positions[1:3, 1:3] = 1
        for i in range(0, 5):
            self.positions[color][max(x - i, 0):min(x + i + 1, self.board_size),
            max(y - i, 0):min(y + i + 1, self.board_size)] += 1
        self.sampling_points[color][x, y] = OPENED

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
        for c in range(LABEL_SIZE):
            print('------------')
            print('分布')
            print('ラベル' + str(c))
            print(self.positions[c])
            print('------------')
            print('サンプリング点')
            print(self.sampling_points[c])
            print('------------')

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

        self.integrate_positions = arr

    def get_optimal_solution(self):
        self.calc_integrate_positions()

        # print(np.amin(integrate_positions))

        x_arr, y_arr = np.where(self.integrate_positions == np.amin(self.integrate_positions))
        # print(x_arr)
        # print(y_arr)

        index = random.randrange(len(x_arr))

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


def main_by_color():
    board_size = 10


if __name__ == '__main__':
    main()
