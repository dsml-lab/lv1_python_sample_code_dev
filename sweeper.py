import numpy as np

'''
点がもつ情報
score
'''

OPENED = 1000.


class Board:

    def __init__(self, board_size):
        self.board_size = board_size
        self.positions = np.zeros((board_size, board_size))
        self.max_position = board_size - 1

    '''
    点を開示
    '''
    def open(self, x, y):
        # x, yを1次元上の値に直す
        # position = self.board_size * y + x

        # 全ての点のx,yからの距離を計算
        # self.positions[1:3, 1:3] = 1
        for i in range(0, 3):

            self.positions[max(x - i, 0):min(x + i + 1, self.board_size), max(y - i, 0):min(y + i + 1, self.board_size)] += 1
        self.positions[x, y] = OPENED

    def init_open(self):
        # あらかじめ４隅に点を打つ
        self.open(x=0, y=0)
        self.open(x=0, y=self.max_position)
        self.open(x=self.max_position, y=0)
        self.open(x=self.max_position, y=self.max_position)

    def print(self):
        print(self.positions)


if __name__ == '__main__':
    b = Board(board_size=10)
    b.print()

    b.init_open()
    b.print()

    b.open(5, 5)
    b.print()
