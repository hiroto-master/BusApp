import cv2
from ultralytics import YOLO
import time
import os
import datetime
import firebase_admin
from firebase_admin import credentials, firestore
import paramiko
import psutil
import threading
import subprocess

# --- 設定 ---
FIREBASE_CRED_PATH = 'serviceAccountKey.json'  # TODO(設定): serviceAccountKey.json
MAC_IP = "<MAC_IP>"  # TODO(設定): <MAC_IP>
MAC_USER = "<MAC_USER>"  # TODO(設定): <MAC_USER>
MAC_PASS = "<MAC_PASS>"  # セキュリティ上、公開鍵認証への移行を強く推奨  TODO(設定): <MAC_PASS>
MAC_DEST_DIR = "/Users/<MAC_USER>/Desktop/received_images"  # TODO(設定): /Users/<MAC_USER>/
MONITOR_INTERVAL = 5  # 監視間隔（秒）

# --- Firebase初期化 ---
try:
    if not firebase_admin._apps:
        cred = credentials.Certificate(FIREBASE_CRED_PATH)
        firebase_admin.initialize_app(cred)
    db = firestore.client()
    print("[Init] Firebase initialized successfully.")
except Exception as e:
    print(f"[Init] Error initializing Firebase: {e}")
    exit()

# --- システム監視関数 ---
def get_cpu_temperature():
    try:
        res = subprocess.check_output(["vcgencmd", "measure_temp"]).decode("utf-8")
        return res.replace("temp=", "").replace("'C\n", "")
    except Exception:
        return "N/A"

def monitor_system(stop_event):
    print("[Monitor] System monitoring thread started.")
    while not stop_event.is_set():
        cpu_usage = psutil.cpu_percent(interval=1)  # 1秒間の平均を取得
        memory_info = psutil.virtual_memory()
        temp = get_cpu_temperature()
        
        timestamp = datetime.datetime.now().strftime('%H:%M:%S')
        print(f"[{timestamp}] [Monitor] CPU: {cpu_usage:>4.1f}% | Mem: {memory_info.percent:>4.1f}% | Temp: {temp}°C")
        
        # 指定間隔待機（stop_eventを監視しながら）
        if stop_event.wait(MONITOR_INTERVAL - 1):
            break
    print("[Monitor] Thread stopped.")

# --- Macへのファイル送信関数 ---
def send_file_to_mac(local_path, mac_path):
    sftp = None
    transport = None
    try:
        transport = paramiko.Transport((MAC_IP, 22))
        transport.connect(username=MAC_USER, password=MAC_PASS)
        sftp = paramiko.SFTPClient.from_transport(transport)
        sftp.put(local_path, mac_path)
        print(f"[Sender] File sent to Mac: {mac_path}")
    except Exception as e:
        print(f"[Sender] Error: {e}")
    finally:
        if sftp: sftp.close()
        if transport: transport.close()

def check_or_create_mac_folder():
    client = None
    try:
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(hostname=MAC_IP, username=MAC_USER, password=MAC_PASS)
        client.exec_command(f'mkdir -p {MAC_DEST_DIR}')
        print("[Init] Mac folder checked/created.")
    except Exception as e:
        print(f"[Init] Error creating directory on Mac: {e}")
    finally:
        if client: client.close()

# --- メイン処理 ---
def main():
    # モデルロード
    print("[Main] Loading YOLOv8 model...")
    model = YOLO('yolov8n.pt')
    print("[Main] Model loaded.")

    # カメラ初期化
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[Main] Error: Could not open video device.")
        return

    # Macフォルダ準備
    check_or_create_mac_folder()

    # 監視スレッド開始
    stop_monitor = threading.Event()
    monitor_thread = threading.Thread(target=monitor_system, args=(stop_monitor,), daemon=True)
    monitor_thread.start()

    try:
        while True:
            start_time = time.time()
            timestamp = datetime.datetime.now()

            # 画像キャプチャ
            ret, frame = cap.read()
            if not ret:
                print("[Main] Error: Failed to capture image.")
                time.sleep(1)
                continue

            # 推論
            print(f"[{timestamp.strftime('%H:%M:%S')}] [Main] Running inference...")
            results = model(frame, verbose=False, conf=0.5)
            
            # 結果処理
            person_count = 0
            result = results[0]
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            names = model.names
            person_indices = [i for i, cls_id in enumerate(class_ids) if names[cls_id] == 'person']
            person_count = len(person_indices)

            print(f"[Main] Detected {person_count} person(s).")

            # 描画・保存・送信
            if person_count > 0:
                for i in person_indices:
                    box = boxes[i].astype(int)
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                    label = f"person: {confidences[i]:.2f}"
                    y = box[1] - 10 if box[1] - 10 > 10 else box[1] + 20
                    cv2.putText(frame, label, (box[0], y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                img_name = f'capture_{timestamp.strftime("%Y%m%d%H%M%S")}.jpg'
                cv2.imwrite(img_name, frame)
                
                # 送信は別スレッドで行い、メインループをブロックしないようにする（オプション）
                # 今回はシンプルにするため同期的に実行
                send_file_to_mac(img_name, os.path.join(MAC_DEST_DIR, img_name))
                os.remove(img_name)

            # Firestore送信
            try:
                db.collection('people_count').add({
                    'count_time': timestamp,
                    'count': person_count
                })
                print(f"[Main] Data uploaded to Firestore.")
            except Exception as e:
                print(f"[Main] Firestore Error: {e}")

            # 次の処理まで待機（処理時間を考慮して調整）
            elapsed = time.time() - start_time
            sleep_time = max(0, 10 - elapsed)
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[Main] Stopping...")
    finally:
        stop_monitor.set()
        monitor_thread.join()
        cap.release()
        print("[Main] Finished.")

if __name__ == '__main__':
    main()
