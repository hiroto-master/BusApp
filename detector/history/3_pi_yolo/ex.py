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

# --- デバッグ設定 ---
DEBUG_SAVE_FRAME = False 
DEBUG_FRAME_PATH = "/tmp"

# --- スレッド間で共有するデータ ---
latest_person_counts = { 0: 0, 2: 0 }
shared_data_lock = threading.Lock()

# --- Firebaseの初期化 ---
try:
    service_account_path = 'serviceAccountKey.json'  # TODO(設定): serviceAccountKey.json
    cred = credentials.Certificate(service_account_path)
    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred)
    db = firestore.client()
    print("[Init] Firebase initialized successfully.")
except Exception as e:
    print(f"[Init] Error initializing Firebase: {e}")
    exit()

# --- CPU・メモリ監視スレッド ---
def monitor_system(initial_time, stop_event):
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [Monitor] 監視スレッドを開始します。")
    psutil.cpu_percent(percpu=True) 
    MONITOR_INTERVAL = 2.0 
    while not stop_event.wait(MONITOR_INTERVAL):
        elapsed_time = time.time() - initial_time
        cpu_list = psutil.cpu_percent(percpu=True)
        mem = psutil.virtual_memory()
        cpu_stats = [f"{v:5.1f}%" for v in cpu_list]
        cpu_string = " | ".join(cpu_stats)
        mem_percent = mem.percent
        log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{log_time}] [Monitor] Time: {int(elapsed_time):>4d}s | CPU: [ {cpu_string} ] | Memory: {mem_percent:5.1f}% Used")
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [Monitor] 監視スレッドを終了します。")

# --- AI処理スレッド (完全同時実行版) ---
def process_camera_task(camera_id, yolo_model, stop_event, save_trigger):
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"[Cam {camera_id}] Error: Could not open video device {camera_id}.")
        return

    print(f"[Cam {camera_id}] カメラ {camera_id} の「完全同時・常時監視」を開始します。")
    try:
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                print(f"[Cam {camera_id}] Error: Failed to capture image.")
                stop_event.wait(0.5) 
                continue
            
            # ★ロックなしで実行★
            results = yolo_model(frame, verbose=False) 
            
            person_count = 0
            result = results[0]
            labels = result.boxes.cls.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            class_names = yolo_model.names
            for i in range(len(labels)):
                if scores[i] > 0.8 and class_names[int(labels[i])] == 'person':
                    person_count += 1

            print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [Cam {camera_id}] Detected {person_count} person(s).")

            with shared_data_lock:
                latest_person_counts[camera_id] = person_count
            
            if save_trigger.is_set():
                 save_trigger.clear()
            
    except Exception as e:
        print(f"[Cam {camera_id}] Error: {e}")
    finally:
        cap.release()
        print(f"[Cam {camera_id}] Released.")

# --- Firebase報告スレッド ---
def firebase_uploader_task(stop_event, interval_seconds=10.0):
    print(f"[Uploader] Firebase報告スレッドを開始します ({interval_seconds}秒間隔)。")
    try:
        while not stop_event.wait(interval_seconds):
            with shared_data_lock:
                c0 = latest_person_counts[0]
                c2 = latest_person_counts[2]
            ts = datetime.datetime.now()
            print(f"[{ts.strftime('%H:%M:%S')}] [Uploader] Report: Cam0={c0}, Cam2={c2}")
            try:
                db.collection('people_count').add({'count_time': ts, 'count': c0, 'camera_id': 0})
                db.collection('people_count').add({'count_time': ts, 'count': c2, 'camera_id': 2})
            except Exception as e:
                print(f"[Uploader] Error: {e}")
    except Exception as e:
        print(f"[Uploader] Error: {e}")

# --- メイン実行部分 ---
if __name__ == "__main__":
    print("[Main] Loading YOLOv9 model...")
    model = YOLO('yolov9c.pt')

    stop_event = threading.Event()
    save_trig_0 = threading.Event()
    save_trig_2 = threading.Event()

    # 監視スレッド
    threading.Thread(target=monitor_system, args=(time.time(), stop_event), daemon=True).start()

    # カメラ0 & 2 を同時に起動
    print("[Main] カメラ0 & 2 を同時に起動します...")
    threading.Thread(target=process_camera_task, args=(0, model, stop_event, save_trig_0), daemon=True).start()
    threading.Thread(target=process_camera_task, args=(2, model, stop_event, save_trig_2), daemon=True).start()

    # 報告スレッド
    threading.Thread(target=firebase_uploader_task, args=(stop_event, 10.0), daemon=True).start()

    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        print("[Main] Stopping...")
        stop_event.set()
