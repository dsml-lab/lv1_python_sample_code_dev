"""
Mnist Color Map
"""
import gym
import numpy as np

from my_clone import LV1_TargetClassifier

ZERO = 0
ONE = 1
TWO = 2
THREE = 3
FOUR = 4
FIVE = 5
SIX = 6
SEVEN = 7
EIGHT = 8
NINE = 9

UNKNOWN = -1


class NumberColorMapEnv(gym.Env):

    def __init__(self, image_size, target):
        """
        Args:
            image_size: 画像のサイズ
        """
        self.image_size = image_size
        self.points = image_size * image_size
        self.action_space = gym.spaces.Discrete(self.points)
        self.state = np.full((self.image_size, self.image_size), -1)
        self.remaining = self.points
        self.target = target

    def step(self, action):
        y = action // self.image_size
        x = action % self.image_size

        observation = self._observe(x, y)

    def reset(self):
        self.state = np.full((self.image_size, self.image_size), -1)
        self.remaining = self.points
        return self.state

    def render(self, mode='human'):
        pass

    def _close(self):
        super().close()

    def _seed(self, seed=None):
        return super().seed(seed)

    def _observe(self, x, y):
        if self.state[x, y] is UNKNOWN:  # 開封済み
            return self.state
        elif self.remaining == 0:  # 残数が0
            return self.state
        else:
            x_normalize = (x + 1) / self.image_size
            y_normalize = (y + 1) / self.image_size

            observation = self.state.copy()
            observation[x, y] = self.target.predict_once(x1=x_normalize, x2=y_normalize)

            self.remaining = self.remaining - 1

            return observation


if __name__ == '__main__':
    target = LV1_TargetClassifier()
    target.load('lv1_targets/classifier_08.png')
    env = NumberColorMapEnv(image_size=4, target=target)
    env.reset()
    print(env.state)
    print(env.points)
    print(env.action_space)
    print(env.state[0, 0])
