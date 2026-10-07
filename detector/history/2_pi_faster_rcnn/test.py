import cv2
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
import time
import os
import datetime
import firebase_admin
from firebase_admin import credentials, firestore
import paramiko # paramikoライブラリを追加

# Firebaseのサービスアカウントキーを初期化
# このファイルはPythonスクリプトと同じディレクトリに配置してください
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

# Macへのファイル送信設定
# --- ご自身の環境に合わせて設定してください ---
MAC_IP = "<MAC_IP>"  # TODO(設定): <MAC_IP>
MAC_USER = "<MAC_USER>"  # TODO(設定): <MAC_USER>
MAC_PASS = "<MAC_PASS>" # パスワードを直接書く代わりにSSHキーの使用を推奨します  TODO(設定): <MAC_PASS>
MAC_DEST_DIR = "/Users/<MAC_USER>/Desktop/received_images" # Macの保存先フォルダ  TODO(設定): /Users/<MAC_USER>/
# -----------------------------------------

def send_file_to_mac(local_path, mac_path):
    """SFTPを使用してMacにファイルを送信する関数"""
    transport = None
    sftp = None
    try:
        transport = paramiko.Transport((MAC_IP, 22))
        transport.connect(username=MAC_USER, password=MAC_PASS)
        sftp = paramiko.SFTPClient.from_transport(transport)
        sftp.put(local_path, mac_path)
        print(f"File sent to Mac: {mac_path}")
    except Exception as e:
        print(f"Error sending file to Mac: {e}")
    finally:
        if sftp:
            sftp.close()
        if transport:
            transport.close()
            
def check_or_create_mac_folder():
    """SSH経由でMac上のフォルダが存在するか確認し、なければ作成する関数"""
    client = None
    try:
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(hostname=MAC_IP, username=MAC_USER, password=MAC_PASS)
        # `mkdir -p` はフォルダが既に存在してもエラーにならない便利なコマンド
        client.exec_command(f'mkdir -p {MAC_DEST_DIR}')
    except Exception as e:
        print(f"Error creating directory on Mac: {e}")
    finally:
        if client:
            client.close()
        
# メインループ
try:
    # プログラム起動時に一度だけMacのフォルダを確認・作成
    check_or_create_mac_folder()

    while True:
        timestamp = datetime.datetime.now()
        
        # 画像をキャプチャ
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            continue
        
        # 画像をPyTorchのテンソルに変換
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        # 推論の実行
        print(f"[{timestamp.strftime('%Y-%m-%d %H:%M:%S')}] Running inference...")
        with torch.no_grad():
            prediction = model([img_tensor])

        # 検出結果の処理
        boxes = prediction[0]['boxes']
        labels = prediction[0]['labels']
        scores = prediction[0]['scores']
        
        person_count = 0
        # 検出した人物のインデックス（何番目の検出結果か）を保存するリスト
        person_indices = []

        # まず人数をカウントし、バウンディングボックス描画対象のインデックスを保存
        for i, (label, score) in enumerate(zip(labels, scores)):
            
            # 【重要】label番号がリストの範囲内にあるかチェックする条件を追加
            if score > 0.8 and label.item() < len(COCO_INSTANCE_CATEGORY_NAMES) and COCO_INSTANCE_CATEGORY_NAMES[label.item()] == 'person':
                person_count += 1
                person_indices.append(i)

        print(f"Detected {person_count} person(s).")

        # 検出された人数が1人以上の場合に、描画、保存、送信を行う
        if person_count > 0:
            # 検出したすべての人物にバウンディングボックスを描画
            for i in person_indices:
                box = boxes[i].detach().numpy().astype(int)
                score = scores[i]
                
                # 緑色の四角形を描画 (BGR形式なので (0, 255, 0) が緑)
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                
                # ラベルとスコア（信頼度）を文字で描画
                label_text = f"person: {score:.2f}"
                cv2.putText(frame, label_text, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 描画後の画像を一時的に保存
            image_filename = f'capture_{timestamp.strftime("%Y%m%d%H%M%S")}.jpg'
            image_path = f'./{image_filename}'
            cv2.imwrite(image_path, frame)
            print(f"Saved annotated image: {image_path}")

            # Macに描画後の写真を送信
            mac_dest_path = os.path.join(MAC_DEST_DIR, image_filename)
            send_file_to_mac(image_path, mac_dest_path)
            
            # 送信後、一時ファイルを削除
            try:
                os.remove(image_path)
                print(f"Deleted temporary image file: {image_filename}")
            except OSError as e:
                print(f"Error deleting file: {e}")
        
        # Firebase Firestoreへのデータアップロード
        doc_data = {
            'count_time': timestamp,
            'count': person_count,
        }
        
        collection_name = 'people_count'
        try:
            db.collection(collection_name).add(doc_data)
            print(f"Uploaded data to Firestore: {person_count} persons detected.")
        except Exception as e:
            print(f"Error uploading data to Firestore: {e}")
        
        # 処理間隔を設定 (例: 10秒)
        time.sleep(10)

except KeyboardInterrupt:
    print("Program terminated by user.")
finally:
    # プログラム終了時にカメラを解放
    cap.release()
    print("Camera released.")
