# YOLOのすゝめ

## YOLOとは？
YOLO（You Only Look Once）は、リアルタイムで物体検出を行うための深層学習アルゴリズムの一種です。YOLOは、入力画像全体を一度に処理し、画像内に存在する複数の物体を同時に検出して分類します。これにより、非常に高速かつ正確な物体検出が可能です。racecarでは瞬時の判断が必要になるため、時間のかかるニューラルネットワークを作るよりもYOLOを使った方が圧倒的に良いです。このリポジトリではYOLOの使い方と説明をします。

## 機械学習の仕組み
機械学習とは機械が学習をすることです(笑)。ただ何も与えずに学習してくれるわけではありません。小学生が答え合わせをせずに算数ドリルをやっても何も身につかないのと一緒です。そこで機械に学習するためのデータを渡してあげます。YOLOに学習してもらう場合には認識する画像とその画像のどの部分に何があるのかをまとめたラベルを一緒に渡します。このリポジトリのdataset構造は以下のようになっています。

```
dataset/
│
├── images/
│   ├── train/
│   ├── val/
│
└── labels/
    ├── train/
    ├── val/
```

imagesとlabelsがあるのにお気づきでしょうか。imagesには画像たちが入っていて、labelsにはそれぞれの画像に呼応しているラベルがtxtファイル形式で保存することになっています。具体例として以下のように配置することが一般的です。

```
dataset/
├── images/
│   ├── train/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   └── val/
│       ├── img3.jpg
│       ├── img4.jpg
├── labels/
    ├── train/
    │   ├── img1.txt
    │   ├── img2.txt
    └── val/
        ├── img3.txt
        ├── img4.txt
```
imagesとlabelsの下にあるtrainとvalはそれぞれ学習用データと検証用データです。学習用データは機械が学習するため、検証用データは機械が学習し終わった時にどのくらい正確なモデルになったかを確かめるためにあります。基本的に検証用データは学習用データのように莫大な量がある必要はなく、一般的には(train:val)は(7:3)や(8:3)くらいの比率にすることが多いです。データセットを分割して一部分をvalに入れるようなイメージです。

labelに入っているtxtファイルは以下の形式で記述します。

```
<class_id> <x_center> <y_center> <width> <height>
```

このようなデータセットは一般的であるためkaggle(データセットのwebサイト)などで探すとたくさん転がっていると思います。

## 環境設定
ここではPythonを使います。Pythonの環境設定は互換性の関係で複雑になることが多いので、condaやvenvなどのパッケージ管理ソフトを使うことを強くお勧めします(筆者の個人的な感想ですがcondaの方がお勧めです)。Pythonは3.8以降のものを使用するといいでしょう。モデルを訓練するために必要なパッケージをインストールしましょう。

```sh
pip install ultralytics opencv-python
```

エラーが出たら

```sh
pip3 install ultralytics opencv-python
```

を実行してみてください。次にデータセットを用意します。データセットの形式は上に書いてありますが、全部自前で用意するにはとても手間がかかりすぎるので例として[text](https://www.kaggle.com/datasets/meeratif/yolo-format-data)を使ってみましょう。

まずはこのリポジトリをクローンしてみましょう。

```sh
git clone https://github.com/motokinakajima/YOLO.git
cd YOLO
```
このサイトに飛んだらサイト上の右上にあるdownloadを押してダウンロードしてみましょう。ダウンロードができたらクローンしたYOLOのファイルの中のdatasetのフォルダを変えてみましょう。元あったファイル構造に沿うようにファイルをコピペしてみてください。.gitkeepと書いてあるファイルは消してしまって構いません(githubの仕様上追加しているものなので)。また、dataset.yamlの内容を変更してみましょう。これはダウンロードした中のcoco128.yamlを参考にして書きます

```yaml
path: ./dataset/images #データセットのディレクトリ
train: ./dataset/images/train #学習用の画像のディレクトリ
val: ./dataset/images/val #検証用の画像のディレクトリ

names: #識別する種類の列挙
  0: gadi
  1: piyari
  2: hamrah
  3: mobile
  4: watch
  5: bili
  6: khuto
```

これで準備は整いました。