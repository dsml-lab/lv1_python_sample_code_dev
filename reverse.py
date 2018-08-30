import numpy as np
import random

np.set_printoptions(suppress=True)
LABEL_SIZE = 10


class OseroBoard:

    def __init__(self, board_size_x: int, board_size_y: int):
        self.board_size_x = board_size_x
        self.board_size_y = board_size_y
        self.positions = np.zeros((LABEL_SIZE, board_size_x, board_size_y))
        self.sampling_points = np.full((LABEL_SIZE, board_size_x, board_size_y), True)
        self.sampling_points_all = np.full((board_size_x, board_size_y), True)
        self.max_position_x = board_size_x - 1
        self.max_position_y = board_size_y - 1

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

    # 点を開示
    def open_once(self, x, y, color):
        self.sampling_points[color][x, y] = False
        self.sampling_points_all[x, y] = False

        for l in range(LABEL_SIZE):
            for x in range(self.board_size_x):
                for y in range(self.board_size_y):
                    self.positions[l][x, y] = self.position_sandwiched(x=x, y=y, color=color)

    def position_sandwiched(self, x, y, color):
        if np.any(self.sampling_points[color][0:x, y]==False) and np.any(self.sampling_points[color][min(x+1, self.max_position_x):self.board_size_x, y]==False):
            return 1

        if np.any(self.sampling_points[color][x, 0:y]==False) and np.any(self.sampling_points[color][x, min(y+1, self.max_position_y):self.board_size_y]==False):
            return 1

        return 0

    def calc_integrate_positions(self):
        arr = np.zeros((self.board_size_x, self.board_size_y))

        for l in range(LABEL_SIZE):
             arr = arr + self.positions[l]

        one_arr = np.ones((self.board_size_x, self.board_size_y))
        arr = np.where(self.sampling_points_all, arr, one_arr)

        return arr

    def get_optimal_solution(self):
        integrate_arr = self.calc_integrate_positions()
        print(integrate_arr)

        max_value = np.amax(integrate_arr)

        if max_value > 1:
            x_arr, y_arr = np.where(integrate_arr == max_value)

            x_y_arr = list(zip(x_arr, y_arr))
            random.shuffle(x_y_arr)

            index = random.randrange(len(x_y_arr))
            select_x, select_y = x_y_arr[index]
            return self.mapping_feature_x_y(select_x, select_y)
        else:
            x_arr, y_arr = np.where(integrate_arr == 0)

            x_y_arr = list(zip(x_arr, y_arr))
            random.shuffle(x_y_arr)

            index = random.randrange(len(x_y_arr))
            select_x, select_y = x_y_arr[index]
            return self.mapping_feature_x_y(select_x, select_y)


if __name__ == '__main__':

    o = OseroBoard(board_size_x=10, board_size_y=10)
    o.open_once(1, 1, 2)
    o.open_once(1, 6, 2)
    o.open_once(1, 1, 2)
    o.open_once(5, 6, 2)

    print(o.positions)

    o.calc_integrate_positions()
