import unittest

from src.sweeper import Board


class SweeperTest(unittest.TestCase):

    def test_mapping_feature_x_y(self):
        board_size = 2
        b = Board(board_size=board_size)

        expected_x = -0.5
        expected_y = 0.5

        actual_x, actual_y = b.mapping_feature_x_y(x=0, y=1)
        self.assertEqual(expected_x, actual_x)
        self.assertEqual(expected_y, actual_y)

    def test_mapping_feature_x_y_4(self):
        board_size = 4
        b = Board(board_size=board_size)

        expected_x = -0.25
        expected_y = 0.25

        actual_x, actual_y = b.mapping_feature_x_y(x=1, y=2)
        self.assertEqual(expected_x, actual_x)
        self.assertEqual(expected_y, actual_y)

    def test_mapping_x_y(self):
        board_size = 2
        b = Board(board_size=board_size)

        expected_x = 0
        expected_y = 1

        actual_x, actual_y = b.mapping_x_y(feature_x=-0.5, feature_y=0.5)
        self.assertEqual(expected_x, actual_x)
        self.assertEqual(expected_y, actual_y)

    def test_mapping_x_y_6(self):
        board_size = 6
        b = Board(board_size=board_size)

        expected_x = 3
        expected_y = 0

        actual_x, actual_y = b.mapping_x_y(feature_x=0.26, feature_y=-0.99)
        self.assertEqual(expected_x, actual_x)
        self.assertEqual(expected_y, actual_y)

    def test_mapping_x_y_12(self):
        board_size = 12
        b = Board(board_size=board_size)

        expected_x = 5
        expected_y = 9

        actual_x, actual_y = b.mapping_x_y(feature_x=-0.08333333333333337, feature_y=0.5833333333333333)
        self.assertEqual(expected_x, actual_x)
        self.assertEqual(expected_y, actual_y)