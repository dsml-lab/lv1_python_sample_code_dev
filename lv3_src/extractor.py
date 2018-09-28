from skimage.feature import local_binary_pattern
import skimage
from skimage.feature import hog
import numpy as np

BINS_SIZE = 256

# 画像特徴抽出器に相当するクラス
# このサンプルコードでは Local Binary Patterns を抽出することにする（skimageを使用）
class LV3FeatureExtractor:

    # 画像 img から抽出量を抽出する
    def extract_lbp(self, img):
        lbp = local_binary_pattern(img, 8, 1, method="uniform")  # local binary pattern
        f, bins = np.histogram(lbp, bins=BINS_SIZE, range=(0, BINS_SIZE-1), density=True)
        return np.asarray(f, dtype=np.float32)

    def extract_hog(self, img):
        orientations = 9
        pixels_per_cell = (8, 8)
        cells_per_block = (3, 3)
        hog = skimage.feature.hog(img, orientations=orientations, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block)
        f, bins = np.histogram(hog, bins=BINS_SIZE, range=(0, BINS_SIZE-1), density=True)
        return np.asarray(f, dtype=np.float32)