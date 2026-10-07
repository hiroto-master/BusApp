import cv2
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
import time
import os
import datetime
import firebase_admin
from firebase_admin import credentials, firestore

# 実行時間帯の設定
# 例: 午前8時から午後5時まで実行したい場合
start_time_str = "08:00"
end_time_str = "17:00"

# Firebaseのサービスアカウントキーを初期化
service_account_path = 'serviceAccountKey.json'  # TODO(設定): serviceAccountKey.json
try:
    cred = credentials.Certificate(service_account_path)
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    print("Firebase initialized successfully.")
except Exception as e:
    print(f"Error initializing Firebase: {e}")
    exit()

# Faster R-CNNモデルのロード
print("Loading Faster R-CNN model...")
weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn_v2(weights=weights, pretrained=True)
model.eval()
print("Model loaded successfully.")

# クラス名の定義
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
    'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

# カメラの初期化
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video device.")
    exit()

# メインループ
try:
    start_h, start_m = map(int, start_time_str.split(':'))
    end_h, end_m = map(int, end_time_str.split(':'))

    while True:
        now = datetime.datetime.now().time()
        start_time = datetime.time(start_h, start_m)
        end_time = datetime.time(end_h, end_m)

        if start_time <= now <= end_time:
            # 実行時間帯内
            print(f"Active: Current time {now.strftime('%H:%M:%S')} is within active period ({start_time_str} - {end_time_str}). Executing...")
            timestamp = datetime.datetime.now()
            image_filename = f'capture_{timestamp.strftime("%Y%m%d%H%M%S")}.jpg'
            image_path = f'./{image_filename}'
            
            # 画像をキャプチャ
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture image.")
                time.sleep(10)
                continue
            
            # 一時的に画像を保存
            cv2.imwrite(image_path, frame)

            # 画像をPyTorchのテンソルに変換
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

            # 推論の実行
            print(f"[{timestamp.strftime('%Y-%m-%d %H:%M:%S')}] Running inference...")
            with torch.no_grad():
                prediction = model([img_tensor])

            # 検出結果の処理と人数カウント
            boxes = prediction[0]['boxes']
            labels = prediction[0]['labels']
            scores = prediction[0]['scores']
            
            person_count = 0
            for label, score in zip(labels, scores):
                if score > 0.8 and COCO_INSTANCE_CATEGORY_NAMES[label.item()] == 'person':
                    person_count += 1

            print(f"Detected {person_count} person(s).")

            # Firebase Firestoreにデータをアップロード
            doc_data = {
                'count_time': firestore.SERVER_TIMESTAMP,
                'count': person_count,
            }
            
            collection_name = 'people_count'
            try:
                db.collection(collection_name).add(doc_data)
                print(f"Uploaded data to Firestore: {person_count} persons detected.")
            except Exception as e:
                print(f"Error uploading data to Firestore: {e}")
            
            # AI処理後の写真を削除
            try:
                os.remove(image_path)
                print(f"Deleted temporary image file: {image_filename}")
            except OSError as e:
                print(f"Error deleting file: {e}")
            
            # 処理間隔を設定 (例: 10秒)
            time.sleep(10)

        else:
            print(f"Waiting: Current time {now.strftime('%H:%M:%S')} is outside of the active period ({start_time_str} - {end_time_str}).")
            # 実行時間外は、1分ごとに時間をチェック
            time.sleep(60)

except KeyboardInterrupt:
    print("Program terminated by user.")
finally:
    # プログラム終了時にカメラを解放
    cap.release()
    print("Camera released.")
