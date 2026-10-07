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
import psutil
import threading
import subprocess
import numpy as np

# --- 設定 ---
FIREBASE_CRED_PATH = 'serviceAccountKey.json'  # TODO(設定): serviceAccountKey.json
MAC_IP = "<MAC_IP>"  # TODO(設定): <MAC_IP>
MAC_USER = "<MAC_USER>"  # TODO(設定): <MAC_USER>
MAC_PASS = "<MAC_PASS>"  # TODO(設定): <MAC_PASS>
MAC_DEST_DIR = "/Users/<MAC_USER>/Desktop/received_images"  # TODO(設定): /Users/<MAC_USER>/
MONITOR_INTERVAL = 5

# COCOクラス名
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

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
    except Exception: return "N/A"

def monitor_system(stop_event):
    print("[Monitor] Thread started.")
    while not stop_event.is_set():
        cpu = psutil.cpu_percent(interval=1)
        mem = psutil.virtual_memory().percent
        temp = get_cpu_temperature()
        ts = datetime.datetime.now().strftime('%H:%M:%S')
        print(f"[{ts}] [Monitor] CPU: {cpu:>4.1f}% | Mem: {mem:>4.1f}% | Temp: {temp}°C")
        if stop_event.wait(MONITOR_INTERVAL - 1): break
    print("[Monitor] Thread stopped.")

# --- Macへの送信関数 ---
def send_file_to_mac(local_path, mac_path):
    sftp, transport = None, None
    try:
        transport = paramiko.Transport((MAC_IP, 22))
        transport.connect(username=MAC_USER, password=MAC_PASS)
        sftp = paramiko.SFTPClient.from_transport(transport)
        sftp.put(local_path, mac_path)
        print(f"[Sender] Sent to Mac: {mac_path}")
    except Exception as e: print(f"[Sender] Error: {e}")
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
        print("[Init] Mac folder checked.")
    except Exception as e: print(f"[Init] Mac folder error: {e}")
    finally: 
        if client: client.close()

# --- メイン処理 ---
def main():
    print("[Main] Loading Faster R-CNN...")
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=weights).eval()
    # GPUが使えるならGPUへ転送 (ラズパイでは通常CPU)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    print(f"[Main] Model loaded on {device}.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[Main] Error: Camera not found.")
        return

    check_or_create_mac_folder()
    stop_monitor = threading.Event()
    threading.Thread(target=monitor_system, args=(stop_monitor,), daemon=True).start()

    try:
        while True:
            start_time = time.time()
            timestamp = datetime.datetime.now()

            ret, frame = cap.read()
            if not ret:
                print("[Main] Capture failed."); time.sleep(1); continue

            # 前処理
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0).to(device)

            # 推論
            print(f"[{timestamp.strftime('%H:%M:%S')}] [Main] Inferencing...")
            with torch.no_grad():
                prediction = model(img_tensor)[0]

            # 後処理
            boxes = prediction['boxes'].cpu().numpy()
            labels = prediction['labels'].cpu().numpy()
            scores = prediction['scores'].cpu().numpy()

            person_indices = [i for i, (label, score) in enumerate(zip(labels, scores))
                              if score > 0.8 and COCO_INSTANCE_CATEGORY_NAMES[label] == 'person']
            person_count = len(person_indices)
            print(f"[Main] Detected {person_count} person(s).")

            if person_count > 0:
                for i in person_indices:
                    box = boxes[i].astype(int)
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                    cv2.putText(frame, f"person: {scores[i]:.2f}", (box[0], box[1]-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                img_name = f'capture_{timestamp.strftime("%Y%m%d%H%M%S")}.jpg'
                cv2.imwrite(img_name, frame)
                send_file_to_mac(img_name, os.path.join(MAC_DEST_DIR, img_name))
                os.remove(img_name)

            try:
                db.collection('people_count').add({'count_time': timestamp, 'count': person_count})
                print("[Main] Uploaded to Firestore.")
            except Exception as e: print(f"[Main] Firestore Error: {e}")

            time.sleep(max(0, 10 - (time.time() - start_time)))

    except KeyboardInterrupt: print("\n[Main] Stopping...")
    finally:
        stop_monitor.set()
        cap.release()
        print("[Main] Finished.")

if __name__ == '__main__':
    main()
