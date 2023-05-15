# OpenCV_Specialized-Exercises-in-Artificial-Intelligence

このリポジトリは、大学の授業で行った画像処理に関するプログラムを集めたものです。なお、参考画像は著作権の関係でカットされています。

## Table of Contents

- [Midterm Project](#Midterm-Project)
- [Final Project](#Final-Project)
- [Cat ears plus Heart rotation](#Cat-ears-plus-Heart-rotation)

## Midterm Project

このプログラムの仕組みを説明する
まずは手の検出は講義資料の応用課題を参考にした。
その後じゃんけんは乱数で手の本数からそれに則したネストで勝敗結果を示す。
因みに乱数はransuで確率の操作が可能である
初期値は本物に則する為に以下の参考文献を基に99.3%で本田圭佑さんが勝つように設定している。
おまけで画像処理の複合をするとのことだったので本田圭佑さんにかつらをつけた。
#### 頑張ったこと

1. Leapmotionのソースが少なかったりするなかでチョキを人差し指と中指の二本の組み合わせだけ反応させたいと思った。
2. 結論からいうと出来ました。for文で指の種類が人差し指のときと中指のときの有無をextendedFingersから着想を得た。
3.  231から240行目で人差し指と中指が認識しているかを確認するコードである。
4. 今回かつらをつけようと思ったらかつらが大きくて顔と被るのでかつらの縦と横のサイズを別々に指定した。
#### 参考文献

- [とある科学の備忘録・【C++/OpenCV】動画のプレイヤーを作成する](https://shizenkarasuzon.hatenablog.com/entry/2020/03/21/000437)
- [Leap SDKで指を検出してみよう（Tracking Hands, Fingers, and Tools）](https://www.buildinsider.net/small/leapmotioncpp/002)
- [勝率なんと99％以上！？ペプシの「本田圭佑じゃんけん」が強すぎる　編集部の総力を挙げて挑戦した結果...](https://www.j-cast.com/2019/04/17355553.html?p=all)



## Final Project
このプログラムは、元となった顔画像を同時に現れる五枚の写真から番号（半角数字）で選び当てるモザイク顔当てゲームです。
17枚なので5回チャレンジ出来ます。sentakuの値を変えると選択肢の個数が変化します。
数字の入力はターミナル入力でお願いします。
#### 使用したクラス、関数

- `PNGOverlay`: アルファチャンネルの貼り付けを行うクラス
- `rand_nodup`: 乱数のリストを作成する関数
- `face_scale`: 顔検出をする関数
- `hconcat`: 複数の画像を横にくっつける関数
- `vconcat`: 複数の画像を縦にくっつける関数
- `stack_create`: 乱数のリストを選択肢の画像に置き換える関数
- `feature_matching`: 渡された二つの画像で特微点マッチングを行う関数
- `check_the_answer`: モザイクの倍率を上げながら特微点マッチングを逐次更新してどれが正解かわかるようにする関数
- `start`: ゲームの説明をする関数
- `main`: ゲームの判定と進行をする関数

#### 頑張ったこと

1. モザイクの倍率を変化させること
2. `cv2.imshow` とユーザー入力を同時に行うと `imshow` が読み込み中のままフリーズする問題の解決
3. 選択肢をタイル上に並べること
4. 特微点マッチングの特微点をどれにするかの選択

#### 参考文献

- [OpenCV基礎プログラミング](https://jellyware.jp/aicorex/contents/out_c04_opencv.html)
- [Python, OpenCVで顔検出と瞳検出（顔認識、瞳認識）](https://note.nkmk.me/python-opencv-face-detection-haar-cascade/)
- [Pythonを用いた画像処理(openCV,skimage)](https://qiita.com/taka_baya/items/453e429b466ffaa702c9)
- [CannyEdgeDetection](https://github.com/kotai2003/CannyEdgeDetection/tree/master/t_module)
- [AI人物素材（ベータ版）](https://www.photo-ac.com/main/genface)
- [【Python】OpenCV 画像を並べて表示する方法　サイズ違い対応版](https://small-onigiri.com/program-220626/)

## Cat ears plus Heart rotation

#### 頑張ったこと

1. 回転させるために三角関数を使う必要があった。
2. 回転するハートの個数を変化できるようにした。

#### 回転する数式

1. x座標の計算式
    $$x = round(cos(i) * w * come_and_go + (x + w/2))$$
これは、極座標系から直交座標系に変換する式です。ここで、iは角度、wは長さ、come_and_goは正または負の値であり、位置の移動を表します。xは、初期位置からの移動距離です。round()関数は四捨五入を行います。

2. y座標の計算式
    $$y = round(sin(i) * h * come_and_go + (y + h/2))$$
これも極座標系から直交座標系に変換する式です。ここで、iは角度、hは長さ、come_and_goは正または負の値であり、位置の移動を表します。yは、初期位置からの移動距離です。round()関数は四捨五入を行います。

なお、NumPyライブラリのnp.sin()とnp.cos()関数は、引数としてラジアン単位を受け取るので、np.radians()関数を使って角度をラジアンに変換しています。また、Pythonのround()関数は四捨五入を行いますが、ここでは数学用の式として整数部分のみを取得するために、floor()関数を使っています。
