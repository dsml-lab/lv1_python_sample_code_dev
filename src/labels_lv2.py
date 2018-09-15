# coding: UTF-8

# クラスラベルIDからラベル名を取得
ID2LNAME = [
    "vehicle",
    "animal",
    "in_the_air",
    "on_water",
    "on_ground",
    "wild_animal",
    "pet_animal",
    "with_wheel"
]

# クラスラベルの種類数
N_LABELS = len(ID2LNAME)

# ラベル名からクラスラベルIDを取得
def LNAME2ID(label_name):
    for i in range(0, N_LABELS):
        if label_name == ID2LNAME[i]:
            return i
    return -1
