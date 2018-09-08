このディレクトリには，レベル1用のサンプルコード（python版）が保存されています．
本サンプルコードは以下の三つのファイルからなります．

 - clone.py
 - evaluation.py
 - labels.py

labels.py は各クラスラベルIDと可視化時の色の対応関係を記述したファイルであり，
evaluation.py はクローン認識器を評価するモジュールを記述したファイルです．
この二つは特に変更の必要はありません．

独自アルゴリズムの実装にあたっては，clone.py を編集します．
このファイル中で主に変更すべきものは以下の二つです．

 - LV1_user_function_sampling() 関数
 - LV1_UserDefinedClassifier クラス

 ※ 必要に応じて他の箇所も変更して頂いて構いません．

前者はターゲット認識器に入力する二次元特徴量をサンプリングする関数です．
後者はクローン認識器を表現するクラスで，以下の二つのメソッドを持ちます．

 - fit()
 - predict()

fit() はクローン認識器を訓練するメソッドであり，
predict() は未知の二次元特徴量を認識するメソッドです．
LV1_UserDefinedClassifier は自由に設計して頂きたく思いますが，
上記二つのメソッドは必ず実装するようにしてください．
（実装しないと evaluation.py が正しく動作しません．）

なお，本サンプルコードの実行には以下のものが必要です．

 - numpy
 - scipy
 - sklearn
 - pillow


# メモ

```
nvidia-docker run -v /media/ando/8b052113-9e75-4fb4-af54-53f93bbe44a7/dsml_public/lv1_python_sample_code:/alcon/ -i -t alcon_lv1_keras:alcon /bin/bash
pip freeze > requirements.txt
```

# 実行方法

```
python my_clone.py lv1_targets/classifier_01.png output/out01_cross.png
```


