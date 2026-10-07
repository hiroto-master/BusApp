import cv2
from ultralytics import YOLO
import time
import os
import datetime
import paramiko
import psutil
import threading
import socket


# --- ★設定: ラズパイB (親機) のIPアドレス ---
PI_B_IP_ADDRESS = "<PI_B_IP_ADDRESS>"  # 親機のIPに合わせて変更  TODO(設定): <PI_B_IP_ADDRESS>
PI_B_PORT = 5000


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


# --- Macへの送信設定 ---
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
        print(f"File sent to Mac: {mac_path}")
        sftp.close()
        transport.close()
    except Exception as e:
        print(f"Error sending file to Mac: {e}")


# --- ★クライアント機能 (Pi Bへ送信) ---
def send_count_to_pi_b(count):
    """検出した人数を親機(Pi B)に送信する"""
    try:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.settimeout(2.0)
        client_socket.connect((PI_B_IP_ADDRESS, PI_B_PORT))
        client_socket.sendall(str(count).encode('utf-8'))
        client_socket.close()
        print(f"[Socket] Sent count '{count}' to Pi B.")
    except Exception as e:
        print(f"[Socket] Failed to send to Pi B: {e}")


# --- カメラ処理 ---
def process_camera_task(camera_id, yolo_model, stop_event):
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

            print(f"[Cam {camera_id}] Detected: {person_count} persons.")

            # ★ここで親機に人数を送信
            send_count_to_pi_b(person_count)

            # Macへ画像送信（元画像）
            original_filename = f'original_camA_{timestamp.strftime("%Y%m%d%H%M%S")}.jpg'
            cv2.imwrite(original_filename, frame)
            send_file_to_mac(original_filename, os.path.join(MAC_DIR_BEFORE, original_filename))
            try:
                os.remove(original_filename)
            except:
                pass

            # Macへ画像送信（枠線入り）
            if person_count > 0:
                annotated_filename = f'annotated_camA_{timestamp.strftime("%Y%m%d%H%M%S")}.jpg'
                annotated_frame = frame.copy()

                for i in person_indices:
                    box = boxes[i].astype(int)
                    cv2.rectangle(annotated_frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

                cv2.imwrite(annotated_filename, annotated_frame)
                send_file_to_mac(annotated_filename, os.path.join(MAC_DIR_AFTER, annotated_filename))
                try:
                    os.remove(annotated_filename)
                except:
                    pass

            stop_event.wait(5.0)

    finally:
        cap.release()


# --- メイン処理 ---
if __name__ == "__main__":
    print("[Main] Loading YOLO model...")
    model = YOLO('best_2.pt')

    stop_event = threading.Event()
    initial_time = time.time()

    threading.Thread(target=monitor_system, args=(initial_time, stop_event), daemon=True).start()
    threading.Thread(target=process_camera_task, args=(0, model, stop_event), daemon=True).start()

    print("[Main] All threads started. Press Ctrl+C to stop.")

    try:
        while not stop_event.is_set():
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[Main] Stopping...")
        stop_event.set()
