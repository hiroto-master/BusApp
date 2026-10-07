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
import socket
import smtplib
import ssl
from email.mime.text import MIMEText
import shutil

# --- ★設定: フォルダパスの自動取得 ---
USER_HOME = os.path.expanduser("~")
BASE_DIR = os.path.join(USER_HOME, 'bus_congestion_project')

# ファイルパス
YOLO_MODEL_PATH = os.path.join(BASE_DIR, 'best_2.pt')
CRED_PATH = os.path.join(BASE_DIR, 'serviceAccountKey.json')  # TODO(設定): serviceAccountKey.json

# --- ★Gmail通知設定 ---
GMAIL_USER = "<GMAIL_USER>"  # TODO(設定): <GMAIL_USER>
GMAIL_APP_PASSWORD = "<GMAIL_APP_PASSWORD>"  # TODO(設定): <GMAIL_APP_PASSWORD>
GMAIL_TO = "<GMAIL_TO>"  # TODO(設定): <GMAIL_TO>

# --- Macへの送信設定 ---
MAC_IP = "<MAC_IP>"  # TODO(設定): <MAC_IP>
MAC_USER = "<MAC_USER>"  # TODO(設定): <MAC_USER>
MAC_PASS = "<MAC_PASS>"  # TODO(設定): <MAC_PASS>
MAC_DIR_BEFORE = "/Users/<MAC_USER>/Desktop/raw_images"  # TODO(設定): /Users/<MAC_USER>/
MAC_DIR_AFTER = "/Users/<MAC_USER>/Desktop/received_images"  # TODO(設定): /Users/<MAC_USER>/

# --- Gmail通知関数 ---
last_email_time = 0
MIN_EMAIL_INTERVAL = 60 

def send_gmail_notify(subject, body, force=False):
    global last_email_time
    current_time = time.time()
    
    if not force and (current_time - last_email_time < MIN_EMAIL_INTERVAL):
        return
    
    if GMAIL_USER == "your_email@gmail.com": return

    try:
        msg = MIMEText(body, "plain", "utf-8")
        msg["Subject"] = subject
        msg["From"] = GMAIL_USER
        msg["To"] = GMAIL_TO

        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(GMAIL_USER, GMAIL_APP_PASSWORD)
            server.send_message(msg)
        print(f"Gmail送信成功: {subject}")
        last_email_time = current_time
    except Exception as e:
        print(f"Gmail送信失敗: {e}")

# --- ネット接続確認 ---
def wait_for_network():
    print("ネットワーク接続待機中...")
    while True:
        try:
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            print("ネットワーク接続確認！")
            # ★起動通知 (独立モードであることを通知)
            print("\n[独立モード] 起動完了！Wi-Fi接続OK。")
            send_gmail_notify("【独立モード】起動成功", "ソケット通信なしの単独モードで起動しました。", force=True)
            return
        except OSError:
            print("接続失敗...2秒後に再試行")
            time.sleep(2)

def monitor_network_thread(stop_event):
    global is_connected
    print("[NetMonitor] ネットワーク監視スレッドを開始しました。")

    while not stop_event.is_set():
        try:
            # GoogleのDNSに接続テスト
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            
            # --- 接続成功時 ---
            if not is_connected:
                # さっきまで切れてたのに繋がった！ (再接続 or 初回接続)
                print("\n[NetMonitor] ★ネットワーク接続を確認！(再接続)")
                is_connected = True
                
                # メール送信 (force=Trueで即時送信)
                send_gmail_notify("【Wi-Fi】接続完了", "ネットワークに接続されました。\nデータ送信を再開します。", force=True)
            
        except OSError:
            # --- 接続失敗時 ---
            if is_connected:
                # さっきまで繋がってたのに切れた！
                print("\n[NetMonitor] ⚠️ ネットワーク切断...")
                is_connected = False
        
        # 10秒おきにチェック
        time.sleep(10)

# --- CPU温度 ---
def get_cpu_temperature():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            temp = float(f.read()) / 1000
        return temp
    except Exception:
        return 0.0

# --- 監視スレッド ---
def monitor_system(initial_time, stop_event):
    print(f"[Monitor] 監視スレッドを開始します。")
    MONITOR_INTERVAL = 2.0 
    while not stop_event.wait(MONITOR_INTERVAL):
        cpu_temp = get_cpu_temperature()
        mem = psutil.virtual_memory()
        log_time = datetime.datetime.now().strftime('%H:%M:%S')
        print(f"[{log_time}] [Monitor] Temp: {cpu_temp:4.1f}°C | Mem: {mem.percent:5.1f}%")

# --- Mac送信 ---
def send_file_to_mac(local_path, mac_path):
    try:
        transport = paramiko.Transport((MAC_IP, 22))
        transport.connect(username=MAC_USER, password=MAC_PASS) 
        sftp = paramiko.SFTPClient.from_transport(transport)
        sftp.put(local_path, mac_path)
        print(f"File sent to Mac: {mac_path}")
        sftp.close()
        transport.close()
    except Exception as e:
        print(f"Error sending file to Mac: {e}")

# --- カメラ処理 (単独動作) ---
def process_camera_task(camera_id, yolo_model, stop_event, db_client):
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        msg = f"[Cam {camera_id}] Error: カメラを開けませんでした。"
        print(msg)
        send_gmail_notify("【独立モード】カメラエラー", msg, force=True)
        return

    print(f"[Cam {camera_id}] Processing started (Standalone Mode).")
    
    try:
        while not stop_event.is_set():
            timestamp = datetime.datetime.now()
            ret, frame = cap.read()
            if not ret:
                time.sleep(1) 
                continue
            
            # 推論
            results = yolo_model(frame, verbose=False)
            result = results[0] 
            
            person_count = 0
            person_indices = [] 
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            labels = result.boxes.cls.cpu().numpy()
            class_names = yolo_model.names 

            for i in range(len(boxes)):
                if scores[i] > 0.5 and class_names[int(labels[i])] == 'person':
                    person_count += 1
                    person_indices.append(i) 

            # ★ここが変更点: 通信相手のカウントを足さず、自分のカウントだけを使う
            total_person_count = person_count
            print(f"Detected: {total_person_count} persons (Standalone)")

            # 画像処理 (Mac送信)
            if person_count > -1: 
                original_filename = f'original_standalone_{timestamp.strftime("%Y%m%d%H%M%S")}.jpg'
                original_temp_path = os.path.join(BASE_DIR, original_filename)
                cv2.imwrite(original_temp_path, frame)
                mac_dest_original = os.path.join(MAC_DIR_BEFORE, original_filename)
                send_file_to_mac(original_temp_path, mac_dest_original)
                try: os.remove(original_temp_path)
                except: pass

                if person_count > 0:
                    annotated_filename = f'annotated_standalone_{timestamp.strftime("%Y%m%d%H%M%S")}.jpg'
                    annotated_temp_path = os.path.join(BASE_DIR, annotated_filename)
                    annotated_frame = frame.copy()
                    for i in person_indices:
                        box = boxes[i].astype(int)
                        cv2.rectangle(annotated_frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                    cv2.imwrite(annotated_temp_path, annotated_frame)
                    mac_dest_annotated = os.path.join(MAC_DIR_AFTER, annotated_filename)
                    send_file_to_mac(annotated_temp_path, mac_dest_annotated)
                    try: os.remove(annotated_temp_path)
                    except: pass

            # Firebase送信 (自分だけのデータを送る)
            try:
                doc_data = {
                    'count_time': timestamp,
                    'count': total_person_count, 
                    'detail': "Standalone Mode (1 Camera)"
                }
                db_client.collection('people_count').add(doc_data)
                print(f"[Firestore] Uploaded: {total_person_count}")
            except Exception as e:
                print(f"[Firestore] Error: {e}")
            
            stop_event.wait(5.0) 
    
    except Exception as e:
        msg = f"[Cam {camera_id}] 例外発生: {e}"
        print(msg)
        send_gmail_notify("【独立モード】プロセスエラー", msg, force=True)
    
    finally:
        cap.release()

if __name__ == "__main__":
    wait_for_network()

    # Firebase初期化
    try:
        cred = credentials.Certificate(CRED_PATH)
        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred)
        db = firestore.client()
        print("[Main] Firebase initialized.")
    except Exception as e:
        msg = f"Firebase初期化エラー: {e}\nPATH: {CRED_PATH}"
        print(msg)
        send_gmail_notify("【独立モード】起動エラー", msg, force=True)
        exit()

    # Macフォルダチェック(失敗しても止まらないようにtryで囲む)
    try:
        check_or_create_mac_folder = lambda: None # ダミー定義(関数定義を省略したため簡易実装が必要なら入れる)
        # ※send_file_to_macの中で接続するのでここは省略可、あるいは前のコードから関数を持ってくる
    except: pass

    print(f"[Main] Loading YOLO model from {YOLO_MODEL_PATH}...")
    try:
        model = YOLO(YOLO_MODEL_PATH) 
        print("[Main] Model loaded.")
    except Exception as e:
        msg = f"モデルロードエラー: {e}\nPATH: {YOLO_MODEL_PATH}"
        print(msg)
        send_gmail_notify("【独立モード】モデルエラー", msg, force=True)
        exit()

    stop_event = threading.Event()
    initial_time = time.time()

    # ソケットサーバーのスレッドは起動しない！
    threading.Thread(target=monitor_system, args=(initial_time, stop_event), daemon=True).start()
    threading.Thread(target=process_camera_task, args=(0, model, stop_event, db), daemon=True).start()
    
    print("[Main] Standalone Mode started. (Ctrl+C to stop)")
    
    try:
        while not stop_event.is_set(): 
            time.sleep(1) 
    except KeyboardInterrupt:
        print("\n[Main] Stopping...")
        stop_event.set()

