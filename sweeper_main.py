from sweeper import Board


def main():
    board_size_x = 10
    board_size_y = 10
    b = Board(board_size_x=board_size_x, board_size_y=board_size_y)
    b.print()

    b.open_once(2, 3, 4)
    b.print()

    b.open_once_colorless(3, 3, 2)
    b.print()

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

    b.calc_integrate_positions()
    b.print()



if __name__ == '__main__':
    main()