from sweeper import Board


def main():
    board_size = 8
    b = Board(board_size=board_size)
    b.print()

    b.open_once_colorless(0, 0)
    b.open_once_colorless(2, 7)

    #b.open_once(3, 3, 4)

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



if __name__ == '__main__':
    main()