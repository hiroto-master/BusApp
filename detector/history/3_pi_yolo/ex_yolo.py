import cv2
# import torch # ultralyticsライブラリが内部で使用しますが、明示的なimportは不要になりました
# import torchvision # 不要になりました
from ultralytics import YOLO # ★YOLOv9 (v8) ライブラリをインポート
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
    
    psutil.cpu_percent(percpu=True) 
    MONITOR_INTERVAL = 2.0 

    while not event.wait(MONITOR_INTERVAL):
        elapsed_time = time.time() - initial_time
        cpu_list = psutil.cpu_percent(percpu=True)
        mem = psutil.virtual_memory()
        cpu_stats = [f"{v:5.1f}%" for v in cpu_list]
        cpu_string = " | ".join(cpu_stats)
        mem_percent = mem.percent
        log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{log_time}] [Monitor] Time: {int(elapsed_time):>4d}s | CPU: [ {cpu_string} ] | Memory: {mem_percent:5.1f}% Used")
    
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [Monitor] 監視スレッドを終了します。")

# --- メインの処理 ---

# Firebaseのサービスアカウントキーを初期化
service_account_path = 'serviceAccountKey.json'  # TODO(設定): serviceAccountKey.json
try:
    cred = credentials.Certificate(service_account_path)
    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred)
    db = firestore.client()
    print("Firebase initialized successfully.")
except Exception as e:
    print(f"Error initializing Firebase: {e}")
    exit()

# ★★★ ここからYOLOv9に変更 ★★★

# YOLOv9モデルのロード
print("Loading YOLOv9 model...")
# 'yolov9c.pt' (compact) はラズパイに適した軽量モデルです。
# 初回実行時に自動でダウンロードされます。
model = YOLO('yolov9c.pt') 
print("Model loaded successfully.")

# ★★★ クラス名の定義は不要になりました ★★★
# (YOLOモデルが 'model.names' として内部に持っています)


# カメラの初期化
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video device.")
    exit()

# Macへのファイル送信設定 (変更なし)
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
    initial_time = time.time()
    monitor_thread = threading.Thread(target=monitor_system, args=(initial_time, event))
    monitor_thread.daemon = True 
    monitor_thread.start()
    # --- 監視スレッドの開始 完了 ---

    while True:
        timestamp = datetime.datetime.now()
        
        # 画像をキャプチャ
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            continue
        
        # ★★★ ここからYOLOv9の推論に変更 ★★★

        # 画像の前処理 (テンソル変換など) はYOLOライブラリが自動で行います
        # (img = ... img_tensor = ... の行は不要)

        # 推論の実行 (OpenCVの 'frame' (BGR形式) をそのまま渡します)
        print(f"[{timestamp.strftime('%Y-%m-%d %H:%M:%S')}] [Inference] Running inference...")
        results = model(frame)

        # 検出結果の処理 (ultralyticsライブラリの形式)
        result = results[0] # 最初の画像の結果を取得
        boxes = result.boxes.xyxy.cpu().numpy() # [x1, y1, x2, y2]
        scores = result.boxes.conf.cpu().numpy() # 信頼度
        labels = result.boxes.cls.cpu().numpy()  # クラスID
        
        person_count = 0
        person_indices = []

        # YOLOが認識するクラス名リストを取得
        class_names = model.names 

        for i in range(len(boxes)):
            label_id = int(labels[i])
            score = scores[i]
            
            # YOLOのCOCOデータセットでは 'person' のクラスIDは 0 です
            if score > 0.8 and label_id in class_names and class_names[label_id] == 'person':
                person_count += 1
                person_indices.append(i)

        print(f"[{timestamp.strftime('%Y-%m-%d %H:%M:%S')}] [Inference] Detected {person_count} person(s).")

        # ★★★ 検出結果の描画 (Faster R-CNN版と同じロジック) ★★★
        if person_count > 0:
            for i in person_indices:
                box = boxes[i].astype(int) # .detach() は不要
                score = scores[i]
                
                # 元の 'frame' に直接描画
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                label_text = f"person: {score:.2f}"
                cv2.putText(frame, label_text, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            image_filename = f'capture_{timestamp.strftime("%Y%m%d%H%M%S")}.jpg'
            image_path = f'./{image_filename}'
            cv2.imwrite(image_path, frame) # 描画済みの 'frame' を保存
            print(f"Saved annotated image: {image_path}")

            mac_dest_path = os.path.join(MAC_DEST_DIR, image_filename)
            send_file_to_mac(image_path, mac_dest_path)
            
            try:
                os.remove(image_path)
                print(f"Deleted temporary image file: {image_filename}")
            except OSError as e:
                print(f"Error deleting file: {e}")
        
        # Firebase Firestoreへのデータアップロード (変更なし)
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
        
        time.sleep(10) # 処理間隔

except KeyboardInterrupt:
    print("Program terminated by user.")
finally:
    # プログラム終了時に監視スレッドを停止
    if 'event' in locals():
        print("Stopping monitor thread...")
        event.set()
        if 'monitor_thread' in locals():
            monitor_thread.join()
        print("Monitor thread stopped.")

    # カメラを解放
    if 'cap' in locals() and cap.isOpened():
        cap.release()
        print("Camera released.")
