# coding: UTF-8

import numpy as np


# 構築したクローン認識器を評価するためのクラス
class LV3_Evaluator:

    def __init__(self, set, extractor):
        # 評価用画像データセット中の全画像から特徴量を抽出して保存しておく
        # 本サンプルコードでは処理時間短縮のため先頭1,000枚のみを用いることにする
        self.samples = []
        for i in range(0, 1000):
            f = set.get_feature(i, extractor)
            self.samples.append((i, f)) # 画像番号と特徴量の組を保存
        self.size = len(self.samples)

    # ターゲット認識器とクローン認識器の出力の一致率（F値）を求める
    #   target: ターゲット認識器
    #   model: クローン認識器
    def calc_accuracy(self, target, model):
        self.target_likelihoods = target.predict_proba(self.samples)
        self.clone_likelihoods = model.predict_proba(self.samples)
        target_labels = self.target_likelihoods >= 0.5
        clone_labels = self.clone_likelihoods >= 0.5
        logical_and_labels = np.logical_and(target_labels, clone_labels)
        r_avg = 0
        p_avg = 0
        f_avg = 0
        for j in range(0, self.size):
            target_labels_sum = np.sum(target_labels[j])
            clone_labels_sum = np.sum(clone_labels[j])
            logical_and_labels = np.sum(logical_and_labels[j])
            if target_labels_sum != 0:
                r = logical_and_labels / target_labels_sum
                r_avg += r
            if clone_labels_sum != 0:
                p = logical_and_labels / clone_labels_sum
                p_avg += p
            r = logical_and_labels / target_labels_sum
            p = logical_and_labels / clone_labels_sum
            if r != 0 or p != 0:
                f = 2 * r * p / (r + p)
                f_avg += f
        r_avg /= self.size
        p_avg /= self.size
        f_avg /= self.size
        return r_avg, p_avg, f_avg
