import numpy as np
import random

'''
点がもつ情報
score
'''

np.set_printoptions(suppress=True)
OPENED = 1.
COLORLESS = -1.


class Board:

    def __init__(self, board_size):
        self.board_size = board_size
        self.positions = np.ones((board_size, board_size))
        self.sampling_points = np.zeros((board_size, board_size))
        self.colors = np.zeros((board_size, board_size))
        self.max_position = board_size - 1

    # 小数をpx値にマッピング
    def mapping_x_y(self, feature_x, feature_y):
        h = self.board_size // 2
        x = int(max(0, min(self.board_size - 1, np.round(h * feature_x + h))))
        y = int(max(0, min(self.board_size - 1, np.round(h - h * feature_y))))

        return x, y

    # px値を少数にマッピング
    def mapping_feature_x_y(self, x, y):
        feature_x = x
        feature_y = y

        return feature_x, feature_y

    def open_once_feature(self, feature_x, feature_y, color):
        x, y = self.mapping_x_y(feature_x=feature_x, feature_y=feature_y)

        return self.open_once(x=x, y=y, color=color)


    # 点を開示
    def open_once(self, x, y, color):
        # x, yを1次元上の値に直す
        # position = self.board_size * y + x

        # 全ての点のx,yからの距離を計算
        # self.positions[1:3, 1:3] = 1
        for i in range(0, 5):
            self.positions[max(x - i, 0):min(x + i + 1, self.board_size),
            max(y - i, 0):min(y + i + 1, self.board_size)] += 1
        self.sampling_points[x, y] = OPENED

        self.colors[x, y] = color

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
        print('------------')
        print('分布')
        print(self.positions)
        print('------------')

        print('------------')
        print('サンプリング点')
        print(self.sampling_points)
        print('------------')

    def get_optimal_solution(self):
        print(np.amin(self.positions))

        print(self.positions == np.amin(self.positions))
        print(np.where(self.positions == np.amin(self.positions)))

        x_arr, y_arr = np.where(self.positions == np.amin(self.positions))
        print(x_arr)
        print(y_arr)

        index = random.randrange(len(x_arr))

        return self.mapping_feature_x_y(x_arr[index], y_arr[index])


def main():
    board_size = 10
    b = Board(board_size=board_size)
    b.print()

    # b.init_open()
    print('初期化')
    b.print()

    b.open_once(2, 3, COLORLESS)
    b.print()

    b.open_once(6, 6, COLORLESS)
    b.print()

    print(b.get_optimal_solution())


def main_by_color():
    board_size = 10




if __name__ == '__main__':
    main()
