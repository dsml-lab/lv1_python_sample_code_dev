import numpy as np

# if __name__ == '__main__':
#     sampling_x = 0
#     sampling_y = 2
#     board_size = 4
#
#     arr = np.zeros((board_size, board_size))
#     arr[sampling_x, sampling_y] = 1
#     print(arr)
#
#     normal_arr = get_distribution(x_size=board_size * 2 + 1, y_size=board_size * 2 + 1)  # 正規分布の2次元配列
#
#     print(normal_arr)
#
#     print(normal_arr[board_size-sampling_x:board_size*2-sampling_x, board_size-sampling_y:board_size*2-sampling_y])

if __name__ == '__main__':
    # 完全一致かどうかを調べたい
    a = np.array([[1, 1, 0]])
    b = np.array([[1, 0, 0]])
    c = np.array([[1, 0, 1]])

    # 完全一致
    # a = np.array([[1, 1, 0]])
    # b = np.array([[1, 1, 0]])
    # c = np.array([[1, 1, 0]])

    # 識別器の個数分の値

    print(a + b + c)
