# YOLO ファインチューニングガイド (Google Colab編)

このガイドは、Roboflowからダウンロードした .zip ファイルを使って、Google ColabでYOLOv8（またはv9, v10）のファインチューニングを行う手順を示します。

## ステップ1: Google Colab の準備

- Google Colabを開く: Google Colab にアクセスし、「ノートブックを新規作成」します。
- GPUを有効にする (最重要):
  - メニューの「ランタイム」 -> 「ランタイムのタイプを変更」をクリックします。
  - 「ハードウェア アクセラレータ」で 「T4 GPU」（またはA100, V100）を選択します。
  - 「保存」を押します。GPUを使わないと、学習に数時間〜数日かかってしまいます。

## ステップ2: YOLO (ultralytics) のインストール

Colabのコードセル（[ ] の部分）に以下のコマンドを入力し、実行ボタン（▶）を押します。

```bash
# ultralytics (YOLOv8, v9, v10) ライブラリをインストール
!pip install ultralytics
```

## ステップ3: Roboflowデータのアップロードと解凍

- .zip ファイルのアップロード:
  - Colabの画面左側にある「フォルダ」アイコンをクリックします。
  - そこに、Roboflowからダウンロードした .zip ファイル（例: Bus-Passenger-Detection-1.zip）をドラッグ＆ドロップしてアップロードします。
  - （ファイルサイズによりますが、数秒〜数分かかります）
- .zip ファイルの解凍: 次のセルで、アップロードした .zip ファイル名を指定して解凍します。

  ```bash
  # ★★★ あなたがアップロードした .zip ファイル名に書き換えてください ★★★
  !unzip "Bus-Passenger-Detection-1.zip" -d dataset
  ```

  - 実行すると、dataset という名前のフォルダが作成され、その中に train, valid, test フォルダや data.yaml が展開されます。

## ステップ4: data.yaml の確認 (ほぼ自動)

data.yaml は「学習データがどこにあるか」をYOLOに教える設計図です。

Roboflowが生成した .zip には、Colab環境用のパスが書かれた data.yaml がすでに入っていることが多いですが、念のため確認・修正します。

```bash
# data.yaml の中身を確認します
!cat dataset/data.yaml
```

【重要】もし data.yaml のパスが間違っていたら

もし、上記の cat コマンドで表示された train: や valid: のパスが /content/dataset/train/images のようになっていない場合（例: ../dataset/train/images のようになっている場合）、以下のセルを実行してパスをColab用に修正します。

```python
import yaml
import os

# Colabの現在の作業ディレクトリを取得
colab_base_dir = "/content" 

data_yaml_path = os.path.join(colab_base_dir, "dataset", "data.yaml")

# YAMLファイルを読み込む
with open(data_yaml_path, 'r') as f:
    data = yaml.safe_load(f)

# パスをColabの絶対パスに修正
data['train'] = os.path.join(colab_base_dir, "dataset", "train", "images")
data['val'] = os.path.join(colab_base_dir, "dataset", "valid", "images")
data['test'] = os.path.join(colab_base_dir, "dataset", "test", "images")

# 'path' キーがもし存在したら削除する (YOLOv8では不要なため)
if 'path' in data:
    del data['path']

# 修正した内容をYAMLファイルに書き戻す
with open(data_yaml_path, 'w') as f:
    yaml.dump(data, f)

print("--- data.yaml を修正しました ---")
!cat dataset/data.yaml
```

## ステップ5: ファインチューニング（学習）の実行

いよいよ学習の開始です。
- model= : 学習の土台（ベース）にするモデル。ラズパイで動かすなら、軽量な yolov8n.pt (YOLOv8 Nano) や yolov9c.pt (YOLOv9 Compact) がおすすめです。
- data= : ステップ4で準備した data.yaml ファイルのパス。
- epochs= : 学習の回数。
  - 推奨: 50 〜 100
  - 50回あれば結果が見えてきます。100回に設定しても、学習が進まなくなれば「Early Stopping」機能で自動的に早めに終了してくれるので安心です。
- imgsz= : 画像サイズ。デフォルトの 640 でOKです。

```python
from ultralytics import YOLO

# 1. ベースとなる軽量モデルをロード (例: YOLOv8 Nano)
#    (存在しない場合は自動でダウンロードされます)
model = YOLO('yolov8n.pt')

# 2. ファインチューニングを実行
results = model.train(
    data='dataset/data.yaml',  # data.yamlファイルのパス
    epochs=50,                 # 学習回数 (最初は50、本番は100で自動停止任せもOK)
    imgsz=640,                 # 画像サイズ
    project='bus_project'      # 結果を保存するプロジェクト名
)

print("--- ファインチューニングが完了しました！ ---")
```

学習が始まると、Epoch 1/50... のように進捗が表示されます。GPUを使っていれば、数十枚のデータなら数分〜数十分で完了します。

## ステップ6: 完成したモデルのダウンロード

学習が完了すると、あなたのバス専用にファインチューニングされたモデル（重みファイル）が作成されています。
- 作成場所: bus_project/train/weights/best.pt

以下のセルを実行して、完成した best.pt をあなたのPC（Mac）にダウンロードします。

```python
from google.colab import files

# 完成したモデルファイル (best.pt) をダウンロード
files.download('bus_project/train/weights/best.pt') 
```
