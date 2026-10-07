import cv2
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
import time
import os
import datetime
import firebase_admin
from firebase_admin import credentials, firestore
import paramiko 
import psutil # CPUやメモリの取得ライブラリ
import threading

# --- CPU・メモリ監視スレッド用の関数 ---
def monitor_system(initial_time, event):
    """
    バックグラウンドでCPUとメモリの使用率を定期的にコンソールに出力する関数
    """
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [Monitor] 監視スレッドを開始します。")
    
    # 最初の呼び出しでベースラインを設定 (interval=Noneのため)
    psutil.cpu_percent(percpu=True) 
    
    # 監視インターバル (秒)
    MONITOR_INTERVAL = 2.0 

    while not event.wait(MONITOR_INTERVAL):
        elapsed_time = time.time() - initial_time
        
        # CPU使用率 (コアごと)
        cpu_list = psutil.cpu_percent(percpu=True)
        # メモリ使用率
        mem = psutil.virtual_memory()

        # --- 出力フォーマット ---
        # 1. CPUリストを整形
        # [10.5, 8.2, ...] -> "[ 10.5% |   8.2% | ... ]"
        cpu_stats = [f"{v:5.1f}%" for v in cpu_list]
        cpu_string = " | ".join(cpu_stats)
        
        # 2. メモリを整形
        mem_percent = mem.percent

        # 3. コンソールに出力
        log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{log_time}] [Monitor] Time: {int(elapsed_time):>4d}s | CPU: [ {cpu_string} ] | Memory: {mem_percent:5.1f}% Used")
    
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [Monitor] 監視スレッドを終了します。")

# --- メインの処理 (以下は元のコードとほぼ同じ) ---

# Firebaseのサービスアカウントキーを初期化
service_account_path = 'serviceAccountKey.json'  # TODO(設定): serviceAccountKey.json
try:
    cred = credentials.Certificate(service_account_path)
    if not firebase_admin._apps: # 既に初期化されていないか確認
        firebase_admin.initialize_app(cred)
    db = firestore.client()
    print("Firebase initialized successfully.")
except Exception as e:
    print(f"Error initializing Firebase: {e}")
    exit()

# Faster R-CNNモデルのロード
print("Loading Faster R-CNN model...")
weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn_v2(weights=weights) # pretrained=Trueはv2では不要
model.eval()
print("Model loaded successfully.")

# クラス名の定義 (省略せずそのまま)
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
MAC_IP = "<MAC_IP>"  # TODO(設定): <MAC_IP>
MAC_USER = "<MAC_USER>"  # TODO(設定): <MAC_USER>
MAC_DEST_DIR = "/Users/<MAC_USER>/Desktop/received_images"  # TODO(設定): /Users/<MAC_USER>/

def send_file_to_mac(local_path, mac_path):
    try:
        transport = paramiko.Transport((MAC_IP, 22))
        transport.connect(username=MAC_USER, password="<MAC_PASSWORD>") # パスワードのハードコーディングに注意  TODO(設定): <MAC_PASSWORD>
        sftp = paramiko.SFTPClient.from_transport(transport)
        sftp.put(local_path, mac_path)
        print(f"File sent to Mac: {mac_path}")
    except Exception as e:
        print(f"Error sending file to Mac: {e}")
    finally:
        if 'sftp' in locals():
            sftp.close()
        if 'transport' in locals():
            transport.close()
            
def check_or_create_mac_folder():
    try:
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(hostname=MAC_IP, username=MAC_USER, password="<MAC_PASSWORD>") # パスワードのハードコーディングに注意  TODO(設定): <MAC_PASSWORD>
        client.exec_command(f'mkdir -p {MAC_DEST_DIR}')
        client.close()
    except Exception as e:
        print(f"Error creating directory on Mac: {e}")
        
# メインループ
try:
    check_or_create_mac_folder()
    
    # --- 監視スレッドの開始 ---
    event = threading.Event()
    initial_time = time.time() # <-- 'inital_time' から 'initial_time' に修正
    # targetに関数名を指定, argsにタプルで引数を渡す
    monitor_thread = threading.Thread(target=monitor_system, args=(initial_time, event))
    monitor_thread.daemon = True # メインスレッドが終了したら一緒に終了する
    monitor_thread.start()
    # --- 監視スレッドの開始 完了 ---

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
        img_tensor = img_tensor.unsqueeze(0) # バッチ次元の追加

        # 推論の実行
        print(f"[{timestamp.strftime('%Y-%m-%d %H:%M:%S')}] [Inference] Running inference...")
        with torch.no_grad():
            prediction = model([img_tensor])

        # 検出結果の処理
        boxes = prediction[0]['boxes']
        labels = prediction[0]['labels']
        scores = prediction[0]['scores']
        
        person_count = 0
        person_indices = []

        for i, (label, score) in enumerate(zip(labels, scores)):
            # score > 0.8 は高すぎる可能性があるので 0.5 などに調整も可
            if score > 0.8 and label.item() < len(COCO_INSTANCE_CATEGORY_NAMES) and COCO_INSTANCE_CATEGORY_NAMES[label.item()] == 'person':
                person_count += 1
                person_indices.append(i)

        print(f"[{timestamp.strftime('%Y-%m-%d %H:%M:%S')}] [Inference] Detected {person_count} person(s).")

        if person_count > 0:
            for i in person_indices:
                box = boxes[i].detach().numpy().astype(int)
                score = scores[i]
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                label_text = f"person: {score:.2f}"
                cv2.putText(frame, label_text, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            image_filename = f'capture_{timestamp.strftime("%Y%m%d%H%M%S")}.jpg'
            image_path = f'./{image_filename}'
            cv2.imwrite(image_path, frame)
            print(f"Saved annotated image: {image_path}")

            mac_dest_path = os.path.join(MAC_DEST_DIR, image_filename)
            send_file_to_mac(image_path, mac_dest_path)
            
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
        
        time.sleep(10)

except KeyboardInterrupt:
    print("Program terminated by user.")
finally:
    # プログラム終了時に監視スレッドを停止
    if 'event' in locals():
        print("Stopping monitor thread...")
        event.set()
        if 'monitor_thread' in locals():
            monitor_thread.join() # スレッドが終了するのを待つ
        print("Monitor thread stopped.")

    # カメラを解放
    if 'cap' in locals() and cap.isOpened():
        cap.release()
        print("Camera released.")
