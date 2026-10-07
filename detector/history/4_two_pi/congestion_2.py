#子機のスクリプト
import cv2
from ultralytics import YOLO 
import time
import os
import datetime
import paramiko 
import psutil
import threading
import socket
import smtplib
import ssl
from email.mime.text import MIMEText

# --- ★設定: ファイルの場所 (絶対パス推奨) ---
BASE_DIR = '/home/raspberrypi/bus_congestion_project'
YOLO_MODEL_PATH = os.path.join(BASE_DIR, 'best_2.pt')

# --- ★設定: 親機(Pi B)の情報 ---
# 固定した親機のIPアドレス
PI_B_IP_ADDRESS = "<PI_B_IP_ADDRESS>"  # TODO(設定): <PI_B_IP_ADDRESS>
PI_B_PORT = 5000

# --- ★Gmail通知設定 ---
GMAIL_USER = "<GMAIL_USER>"        # 送信元  TODO(設定): <GMAIL_USER>
GMAIL_APP_PASSWORD = "<GMAIL_APP_PASSWORD>"      # アプリパスワード  TODO(設定): <GMAIL_APP_PASSWORD>
GMAIL_TO = "<GMAIL_TO>"          # 送信先  TODO(設定): <GMAIL_TO>


# --- グローバル変数 ---
first_connection_success = False  # 初回接続成功フラグ

# --- Gmail通知関数 (連続送信防止付き) ---
last_email_time = 0
MIN_EMAIL_INTERVAL = 60 

def send_gmail_notify(subject, body, force=False):
    global last_email_time
    current_time = time.time()
    
    if not force and (current_time - last_email_time < MIN_EMAIL_INTERVAL):
        return

    if GMAIL_USER == "your_email@gmail.com":
        return

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
            
            # ★起動通知
            print("\n[子機] 起動完了！Wi-Fi接続。")
            send_gmail_notify("【子機】起動成功", "子機がWi-Fiに接続しました。\n処理を開始します。", force=True)
            return
        except OSError:
            print("接続失敗...2秒後に再試行")
            time.sleep(2)

# --- CPU温度を取得する関数 ---
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
        print(f"[Monitor] Temp: {cpu_temp:4.1f}°C | Mem: {mem.percent:5.1f}%")

# --- クライアント機能 (Pi Bへ送信) ---
def send_count_to_pi_b(count):
    global first_connection_success
    try:
        # ソケット接続
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.settimeout(2.0)
        
        # 接続エラーが出てもプログラムを落とさない
        try:
            client_socket.connect((PI_B_IP_ADDRESS, PI_B_PORT))
        except (socket.timeout, ConnectionRefusedError, OSError):
            print(f"[Socket] 親機({PI_B_IP_ADDRESS})が見つかりません。スキップ。")
            client_socket.close()
            return # 今回は諦めて次のループへ

        # データ送信
        client_socket.sendall(str(count).encode('utf-8'))
        client_socket.close()
        
        print(f"[Socket] 親機へ送信成功: {count}人")

        # 初回成功通知
        if not first_connection_success:
            first_connection_success = True
            msg = f"親機(Pi B)へのデータ送信に成功しました。\nTarget IP: {PI_B_IP_ADDRESS}"
            send_gmail_notify("【Pi 子機】親機接続成功", msg, force=True)

    except Exception as e:
        print(f"[Socket] 予期せぬエラー: {e}")

def monitor_network_thread(stop_event):
    global is_connected
    print("[NetMonitor] ネットワーク監視スレッドを開始しました。")

    while not stop_event.is_set():
        try:
            # GoogleのDNSに接続テスト
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            
            # --- 接続成功時 ---
            if not is_connected:
                # 再接続 or 初回接続
                print("\n[NetMonitor] ネットワーク接続を確認(再接続)")
                is_connected = True
                
                # メール送信 (force=Trueで即時送信)
                send_gmail_notify("【Wi-Fi】接続完了", "ネットワークに接続されました。\nデータ送信を再開します。", force=True)
            
        except OSError:
            # --- 接続失敗時 ---
            if is_connected:
                # さっきまで繋がってたのに切れた！
                print("\n[NetMonitor] ネットワーク切断...")
                is_connected = False
        
        # 10秒おきにチェック
        time.sleep(10)

# --- カメラ処理 ---
def process_camera_task(camera_id, yolo_model, stop_event):
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        msg = f"[Cam {camera_id}] Error: カメラを開けませんでした。"
        print(msg)
        send_gmail_notify("【Pi 子機】カメラエラー", msg, force=True)
        return

    print(f"[Cam {camera_id}] Processing started.")
    
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
            
            # 人数カウント
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

            print(f"[Cam {camera_id}] Detected: {person_count} persons.")

            # ★親機に人数を送信
            send_count_to_pi_b(person_count)

            
            stop_event.wait(5.0) 
    
    except Exception as e:
        msg = f"カメラスレッドでエラー: {e}"
        print(msg)
        send_gmail_notify("【Pi 子機】プロセスエラー", msg, force=True)

    finally:
        cap.release()

if __name__ == "__main__":
    # 1. ネット接続待機
    wait_for_network()

    print(f"[Main] Loading YOLO model from {YOLO_MODEL_PATH}...")
    try:
        model = YOLO(YOLO_MODEL_PATH) 
    except Exception as e:
        msg = f"モデルロードエラー: {e}"
        print(msg)
        send_gmail_notify("【Pi 子機】モデルエラー", msg, force=True)
        exit()

    stop_event = threading.Event()
    initial_time = time.time()

    # スレッド起動
    threading.Thread(target=monitor_system, args=(initial_time, stop_event), daemon=True).start()
    threading.Thread(target=process_camera_task, args=(0, model, stop_event), daemon=True).start()
    
    print("[Main] Pi 子機 Client started. Press Ctrl+C to stop.")
    
    try:
        while not stop_event.is_set(): 
            time.sleep(1) 
    except KeyboardInterrupt:
        print("\n[Main] Stopping...")
        stop_event.set()
