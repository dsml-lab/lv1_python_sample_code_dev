import numpy as np

from PIL import ImageFilter, Image


# 色を判定して、白または黒を返す
# 引数のthresholdが閾値
def select_color(threshold, color):
    # r,g,bの平均を求める
    mean = np.array(color).mean(axis=0)
    if mean > threshold:
        return 255, 255, 255
    else:
        return 0, 0, 0


# 2値化した画像を返すメソッド
def to_bin(img):
    w, h = img.size
    bin_img = Image.new('RGB', (w, h))
    threshold = 0

    # select_colorメソッドを使って塗る色を決めながら、新しい画像を作っていく
    np.array(
        [[bin_img.putpixel((x, y), select_color(threshold, img.getpixel((x, y)))) for x in range(w)] for y in range(h)])
    return bin_img


def filter_edge(img):
    filtered = img.filter(ImageFilter.FIND_EDGES)
    gray = to_bin(filtered)

    return gray


# if __name__ == '__main__':
#     target = LV1_TargetClassifier()
#     target.load('lv1_targets/classifier_01.png')
#
#     filter_edge(target.img)
