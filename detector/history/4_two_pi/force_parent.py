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


# --- グローバル変数: データ共有 ---
shared_data = {
    "local_count": 0,   # ラズパイB自身の検出数
    "remote_count": 0   # Pi A から受信した人数
}


# --- CPU温度の取得 ---
def get_cpu_temperature():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            temp = float(f.read()) / 1000
        return temp
    except Exception:
        return 0.0


# --- システム監視スレッド ---
def monitor_system(initial_time, stop_event):
    print("[Monitor] 監視スレッドを開始します。")

    psutil.cpu_percent(percpu=True)
    MONITOR_INTERVAL = 2.0

    while not stop_event.wait(MONITOR_INTERVAL):
        elapsed_time = time.time() - initial_time

        cpu_list = psutil.cpu_percent(percpu=True)
        mem = psutil.virtual_memory()
        cpu_temp = get_cpu_temperature()

        cpu_stats = [f"{v:5.1f}%" for v in cpu_list]
        cpu_string = " | ".join(cpu_stats)

        total = shared_data["local_count"] + shared_data["remote_count"]
        count_status = f"Total: {total} (Local: {shared_data['local_count']}, Remote: {shared_data['remote_count']})"

        log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{log_time}] [Monitor] Temp: {cpu_temp:4.1f}°C | {count_status} | Mem: {mem.percent:5.1f}%")


# --- Macへ送信 ---
MAC_IP = "<MAC_IP>"  # TODO(設定): <MAC_IP>
MAC_USER = "<MAC_USER>"  # TODO(設定): <MAC_USER>
MAC_DIR_BEFORE = "/Users/<MAC_USER>/Desktop/raw_images"  # TODO(設定): /Users/<MAC_USER>/
MAC_DIR_AFTER = "/Users/<MAC_USER>/Desktop/received_images"  # TODO(設定): /Users/<MAC_USER>/


def send_file_to_mac(local_path, mac_path):
    try:
        transport = paramiko.Transport((MAC_IP, 22))
        transport.connect(username=MAC_USER, password="<MAC_PASSWORD>")  # TODO(設定): <MAC_PASSWORD>
        sftp = paramiko.SFTPClient.from_transport(transport)
        sftp.put(local_path, mac_path)
        print(f"[Mac] File sent: {mac_path}")

    except Exception as e:
        print(f"[Mac] Error sending file: {e}")

    finally:
        if 'sftp' in locals():
            sftp.close()
        if 'transport' in locals():
            transport.close()


def check_or_create_mac_folder():
    client = None
    try:
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(hostname=MAC_IP, username=MAC_USER, password="<MAC_PASSWORD>")  # TODO(設定): <MAC_PASSWORD>

        client.exec_command(f'mkdir -p {MAC_DIR_BEFORE}')
        client.exec_command(f'mkdir -p {MAC_DIR_AFTER}')

        print("[Mac] Directories checked/created")

    except Exception as e:
        print(f"[Mac] Error creating directory: {e}")

    finally:
        if client:
            client.close()


# --- サーバー: Pi A から受信 ---
def start_server_thread(stop_event):
    HOST = '0.0.0.0'
    PORT = 5000

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    try:
        server_socket.bind((HOST, PORT))
        server_socket.listen(1)
        server_socket.settimeout(1.0)

        print(f"[Server] Listening on port {PORT}...")

        while not stop_event.is_set():
            try:
                conn, addr = server_socket.accept()
                with conn:
                    data = conn.recv(1024)
                    if data:
                        try:
                            count_from_a = int(data.decode('utf-8'))
                            shared_data["remote_count"] = count_from_a
                            print(f"[Server] Received from Pi A: {count_from_a}")

                        except ValueError:
                            print(f"[Server] Invalid data received: {data}")

            except socket.timeout:
                continue

    finally:
        server_socket.close()
        print("[Server] Socket closed")


# --- カメラ処理スレッド ---
def process_camera_task(camera_id, yolo_model, stop_event, db_client):
    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        print(f"[Cam {camera_id}] Error: Could not open video")
        return

    print(f"[Cam {camera_id}] Started camera thread")

    try:
        while not stop_event.is_set():
            timestamp = datetime.datetime.now()
            ret, frame = cap.read()

            if not ret:
                print(f"[Cam {camera_id}] Capture failed")
                time.sleep(1)
                continue

            # 推論
            results = yolo_model(frame, verbose=False)
            result = results[0]

            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            labels = result.boxes.cls.cpu().numpy()
            class_names = yolo_model.names

            # 人カウント
            person_count = 0
            person_indices = []

            for i in range(len(boxes)):
                label_id = int(labels[i])
                if scores[i] > 0.5 and class_names[label_id] == 'person':
                    person_count += 1
                    person_indices.append(i)

            shared_data["local_count"] = person_count
            total_person_count = shared_data["local_count"] + shared_data["remote_count"]

            print(f"[Cam {camera_id}] Local={person_count}  Remote={shared_data['remote_count']}  Total={total_person_count}")

            # --- 元画像送信 ---
            original_filename = f'original_cam{camera_id}_{timestamp.strftime("%Y%m%d%H%M%S")}.jpg'
            cv2.imwrite(original_filename, frame)
            send_file_to_mac(original_filename, os.path.join(MAC_DIR_BEFORE, original_filename))
            os.remove(original_filename)

            # --- 枠付き画像 ---
            if person_count > 0:
                annotated_filename = f'annotated_cam{camera_id}_{timestamp.strftime("%Y%m%d%H%M%S")}.jpg'
                annotated_frame = frame.copy()

                for i in person_indices:
                    box = boxes[i].astype(int)
                    score = scores[i]
                    cv2.rectangle(annotated_frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                    cv2.putText(annotated_frame, f"person {score:.2f}",
                                (box[0], box[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                cv2.imwrite(annotated_filename, annotated_frame)
                send_file_to_mac(annotated_filename, os.path.join(MAC_DIR_AFTER, annotated_filename))
                os.remove(annotated_filename)

            # --- Firestoreにアップロード ---
            doc_data = {
                'count_time': timestamp,
                'count': total_person_count
            }

            try:
                db_client.collection('people_count').add(doc_data)
                print(f"[Firestore] Uploaded Total={total_person_count}")
            except Exception as e:
                print(f"[Firestore] Upload error: {e}")

            stop_event.wait(5.0)

    finally:
        cap.release()
        print(f"[Cam {camera_id}] Released camera")


# --- メイン ---
if __name__ == "__main__":
    print("[Main] Initializing Firestore...")
    cred = credentials.Certificate("serviceAccountKey.json")
    firebase_admin.initialize_app(cred)
    db_client = firestore.client()

    print("[Main] Loading YOLO model...")
    model = YOLO("best_2.pt")

    check_or_create_mac_folder()

    stop_event = threading.Event()
    initial_time = time.time()

    # 監視スレッド
    threading.Thread(target=monitor_system, args=(initial_time, stop_event), daemon=True).start()

    # サーバスレッド（Pi A → Pi B）
    threading.Thread(target=start_server_thread, args=(stop_event,), daemon=True).start()

    # カメラスレッド
    threading.Thread(target=process_camera_task, args=(0, model, stop_event, db_client), daemon=True).start()

    print("[Main] System running. Press Ctrl+C to stop.")

    try:
        while not stop_event.is_set():
            time.sleep(1)
    except KeyboardInterrupt:
        print("[Main] Stopping...")
        stop_event.set()
