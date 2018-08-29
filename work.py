import numpy as np


def get_distribution(size):
    def calc(x, y):
        r = x ** 2 + y ** 2
        return int(np.exp(-r) * (size/2))

    xn = size
    x0 = np.linspace(-2, 2, xn)
    x1 = np.linspace(-2, 2, xn)
    arr = np.zeros((len(x0), len(x1)))
    for i0 in range(xn):
        for i1 in range(xn):
            arr[i1, i0] = calc(x0[i0], x1[i1])

    return arr


if __name__ == '__main__':
    sampling_x = 0
    sampling_y = 2
    board_size = 4

    arr = np.zeros((board_size, board_size))
    arr[sampling_x, sampling_y] = 1
    print(arr)

    normal_arr = get_distribution(size=board_size * 2 + 1)  # 正規分布の2次元配列

    print(normal_arr)

    print(normal_arr[board_size-sampling_x:board_size*2-sampling_x, board_size-sampling_y:board_size*2-sampling_y])
