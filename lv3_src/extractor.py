from skimage.feature import local_binary_pattern
import numpy as np

BINS_SIZE = 256

# 画像特徴抽出器に相当するクラス
# このサンプルコードでは Local Binary Patterns を抽出することにする（skimageを使用）
class LV3FeatureExtractor:

    # 画像 img から抽出量を抽出する
    def extract(self, img):
        lbp = local_binary_pattern(img, 8, 1, method="uniform")  # local binary pattern
        f, bins = np.histogram(lbp, bins=BINS_SIZE, range=(0, BINS_SIZE-1), density=True)
        return np.asarray(f, dtype=np.float32)