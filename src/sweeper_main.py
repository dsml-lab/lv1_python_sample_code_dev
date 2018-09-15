from src.sweeper import Board


def main():
    board_size_x = 5
    board_size_y = 5
    b = Board(board_size_x=board_size_x, board_size_y=board_size_y)
    b.info()

    b.open_once(0, 0, 8)
    b.calc_integrate_positions()
    b.info()

    b.open_once(3, 3, 4)
    b.calc_integrate_positions()
    b.info()

    b.open_once(4, 4, 8)
    b.calc_integrate_positions()
    b.info()


    for i in range(1, 5):
        x, y = b.get_optimal_solution()
        x, y = b.mapping_x_y(x, y)
        b.open_once(x, y, 4)
        b.info()

    #b.open_once(3, 3, 4)

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


if __name__ == '__main__':
    main()