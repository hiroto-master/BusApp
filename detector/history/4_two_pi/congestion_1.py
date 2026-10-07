#親機のスクリプト
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
import shutil

# --- グローバル変数: データを共有する辞書 ---
shared_data = {
    "local_count": 0,   # ラズパイB自身の検出数
    "remote_count": 0   # ラズパイAから送られてきた数
}

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
    psutil.cpu_percent(percpu=True) 
    MONITOR_INTERVAL = 1.0 

    while not stop_event.wait(MONITOR_INTERVAL):
        elapsed_time = time.time() - initial_time
        cpu_temp = get_cpu_temperature()
        mem = psutil.virtual_memory()
        
        total = shared_data["local_count"] + shared_data["remote_count"]
        count_status = f"Total: {total} (Local: {shared_data['local_count']}, Remote: {shared_data['remote_count']})"

        log_time = datetime.datetime.now().strftime('%H:%M:%S')
        print(f"[{log_time}] [Monitor] Temp: {cpu_temp:4.1f}°C | {count_status} | Mem: {mem.percent:5.1f}%")




# --- サーバー機能 (子機からの受信) ---
def start_server_thread(stop_event):
    #<IP_ADDRESS>
    HOST = '<IP_ADDRESS>'  # TODO(設定): <IP_ADDRESS>
    PORT = 5000 
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    try:
        server_socket.bind((HOST, PORT))
        server_socket.listen(1)
        print(f"[Server] Listening for Pi A on port {PORT}...")
        server_socket.settimeout(1.0)

        while not stop_event.is_set():
            try:
                conn, addr = server_socket.accept()
                with conn:
                    data = conn.recv(1024)
                    if data:
                        try:
                            count_from_a = int(data.decode('utf-8'))
                            shared_data["remote_count"] = count_from_a
                            print(f"[Server] Received from Pi A: {count_from_a} persons")
                        except ValueError:
                            pass
            except socket.timeout:
                continue
            except Exception as e:
                print(f"[Server] Error: {e}")
    finally:
        server_socket.close()

# --- カメラ処理 ---
def process_camera_task(camera_id, yolo_model, stop_event, db_client):
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"[Cam {camera_id}] Error: Could not open video device.")
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

            # データを更新
            shared_data["local_count"] = person_count
            total_person_count = shared_data["local_count"] + shared_data["remote_count"]

            # Firebaseへ合計値を送信
            try:
                doc_data = {
                    'count_time': timestamp,
                    'count': total_person_count, 
                    'details': {
                        'pi_b_local': shared_data["local_count"],
                        'pi_a_remote': shared_data["remote_count"]
                    }
                }
                db_client.collection('people_count').add(doc_data)
                print(f"[Firestore] Uploaded Total: {total_person_count} (A:{shared_data['remote_count']} + B:{person_count})")
            except Exception as e:
                print(f"[Firestore] Error: {e}")
            
            stop_event.wait(5.0) 
    
    finally:
        cap.release()

if __name__ == "__main__":
    # Firebase初期化
    try:
        cred = credentials.Certificate('serviceAccountKey.json')  # TODO(設定): serviceAccountKey.json
        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred)
        db = firestore.client()
        print("[Main] Firebase initialized.")
    except Exception as e:
        print(f"[Main] Error initializing Firebase: {e}")
        exit()
    
    print("[Main] Loading YOLO model...")
    model = YOLO('Yolo11l.pt') 

    stop_event = threading.Event()
    initial_time = time.time()

    # スレッド起動
    threading.Thread(target=monitor_system, args=(initial_time, stop_event), daemon=True).start()
    threading.Thread(target=start_server_thread, args=(stop_event,), daemon=True).start()
    threading.Thread(target=process_camera_task, args=(0, model, stop_event, db), daemon=True).start()
    
    print("[Main] All threads started. Press Ctrl+C to stop.")
    
    try:
        while not stop_event.is_set(): 
            time.sleep(1) 
    except KeyboardInterrupt:
        print("\n[Main] Stopping...")
        stop_event.set()
