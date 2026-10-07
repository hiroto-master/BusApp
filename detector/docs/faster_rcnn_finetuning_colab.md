# Faster R-CNN ファインチューニングガイド (Google Colab編)

このガイドは、Roboflowのアノテーションデータを使って、PyTorchの「Faster R-CNN」モデルをあなたのバス乗客検知用に再学習（ファインチューニング）させる手順です。

## ステップ1: Roboflowからデータを再エクスポートする

Faster R-CNNはYOLO形式（.txt）ではなく、**Pascal VOC形式（.xml）**の方が扱いやすいです。
- Roboflowのプロジェクトページを開きます。
- 「Generate」タブ（またはDatasetタブ）に行き、**「Export」**ボタンを押します。
- Format (形式) で Pascal VOC を選択してください。（ここが重要！）
- 「Continue」を押し、**「Download zip to computer」**を選んで、PCに .zip ファイルをダウンロードします。

## ステップ2: Google Colabの準備

- Google Colab で「ノートブックを新規作成」します。
- メニューの「ランタイム」→「ランタイムのタイプを変更」で 「T4 GPU」 を選びます。
- 画面左のフォルダアイコンを開き、ダウンロードした .zip ファイル（例: Bus-Passenger-PascalVOC.zip）をアップロードします。

## ステップ3: 学習用コードの実行

以下の長いコードを、Colabのセルに貼り付けて実行してください。

このコードは、「データの解凍」から「学習」、「モデルの保存」までを一気に行います。

```python
import os
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from PIL import Image
import xml.etree.ElementTree as ET
import zipfile

# --- 1. データの解凍 ---
# ★★★ あなたのアップロードしたzipファイル名に変えてください ★★★
zip_filename = "Bus-Passenger-PascalVOC.zip" 

if os.path.exists(zip_filename):
    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        zip_ref.extractall("dataset")
    print("解凍完了！")
else:
    print("エラー: zipファイルが見つかりません。アップロードしましたか？")

# --- 2. データセットクラスの定義 (XMLを読み込む機能) ---
class BusDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        # 画像ファイルとXMLファイルをソートして取得
        self.imgs = list(sorted(os.listdir(os.path.join(root, "train"))))
        # XMLファイルだけフィルタリング
        self.imgs = [f for f in self.imgs if f.endswith('.jpg')]
        self.xmls = [f.replace('.jpg', '.xml') for f in self.imgs]

    def __getitem__(self, idx):
        # 画像とXMLのパス
        img_path = os.path.join(self.root, "train", self.imgs[idx])
        xml_path = os.path.join(self.root, "train", self.xmls[idx])

        # 画像を開く
        img = Image.open(img_path).convert("RGB")
        
        # XMLをパースして座標を取得
        tree = ET.parse(xml_path)
        root = tree.getroot()
        boxes = []
        for obj in root.findall('object'):
            name = obj.find('name').text
            if name == 'person': # personだけを取得
                bbox = obj.find('bndbox')
                xmin = float(bbox.find('xmin').text)
                ymin = float(bbox.find('ymin').text)
                xmax = float(bbox.find('xmax').text)
                ymax = float(bbox.find('ymax').text)
                boxes.append([xmin, ymin, xmax, ymax])

        # テンソルに変換
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # ラベル (person=1, 背景=0)
        labels = torch.ones((len(boxes),), dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms:
            img = F.to_tensor(img)

        return img, target

    def __len__(self):
        return len(self.imgs)

# --- 3. モデルの準備 ---
def get_model(num_classes):
    # 学習済みのFaster R-CNN (ResNet50) をロード
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    
    # 分類器の入力特徴量数を取得
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # 分類器を、今回のクラス数用に置き換え (背景 + person = 2クラス)
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# --- 4. 学習の実行 ---
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"使用デバイス: {device}")

# データセットの準備
dataset = BusDataset('dataset', transforms=True)
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=2, shuffle=True, num_workers=2,
    collate_fn=lambda x: tuple(zip(*x)) # 異なるサイズの画像をまとめるおまじない
)

# モデルとオプティマイザの設定
num_classes = 2 # 背景(0) + person(1)
model = get_model(num_classes)
model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# 学習ループ (Epoch数はここで変更可能)
num_epochs = 10 

print("学習開始！")
for epoch in range(num_epochs):
    model.train()
    i = 0
    for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if i % 10 == 0:
            print(f"Epoch: {epoch+1}, Step: {i}, Loss: {losses.item():.4f}")
        i += 1
    
    lr_scheduler.step()
    print(f"Epoch {epoch+1} 完了")

print("学習終了！モデルを保存します...")

# --- 5. モデルの保存 ---
# どこでも使えるように、state_dict (重みデータ) だけを保存します
torch.save(model.state_dict(), "fasterrcnn_bus.pth")
print("保存完了: fasterrcnn_bus.pth")
```

## ステップ4: モデルのダウンロード

学習が終わると、Colabのファイル一覧に fasterrcnn_bus.pth というファイルが出来上がります。

これを右クリックしてダウンロードしてください。

## ステップ5: ラズパイ（PC）での使い方

ダウンロードした fasterrcnn_bus.pth を、async_camera_processing.py と同じフォルダに置きます。

そして、async_camera_processing.py の load_faster_rcnn_model 関数を、以下のように少し書き換えることで、学習済みモデルを読み込めます。

```python
def load_faster_rcnn_model():
    # ... (前略) ...
    
    # 1. まず空のモデルを作る (クラス数は学習時と同じ 2 にする)
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None) # 重みは後で入れるのでNone
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2) # 背景+person=2
    
    # 2. 学習した重みファイル (.pth) をロードする
    # map_locationは、GPUで学習したものをCPUで動かす場合に必要
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load("fasterrcnn_bus.pth", map_location=device))
    
    model.eval()
    model.to(device)
    return model, device
```

これで、あなたがアノテーションしたデータを使って賢くなったFaster R-CNNが動きます！
