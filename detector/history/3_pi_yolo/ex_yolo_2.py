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

# --- CPU・メモリ監視スレッド用の関数 (変更なし) ---
def monitor_system(initial_time, stop_event):
    """
    バックグラウンドでCPUとメモリの使用率を定期的にコンソールに出力する関数
    """
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

# --- Macへのファイル送信設定 (変更なし) ---
MAC_IP = "<MAC_IP>"  # TODO(設定): <MAC_IP>
MAC_USER = "<MAC_USER>"  # TODO(設定): <MAC_USER>
MAC_DEST_DIR = "/Users/<MAC_USER>/Desktop/received_images"  # TODO(設定): /Users/<MAC_USER>/

def send_file_to_mac(local_path, mac_path):
    try:
        transport = paramiko.Transport((MAC_IP, 22))
        transport.connect(username=MAC_USER, password="<MAC_PASSWORD>")  # TODO(設定): <MAC_PASSWORD>
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
        client.connect(hostname=MAC_IP, username=MAC_USER, password="<MAC_PASSWORD>")  # TODO(設定): <MAC_PASSWORD>
        client.exec_command(f'mkdir -p {MAC_DEST_DIR}')
        client.close()
    except Exception as e:
        print(f"Error creating directory on Mac: {e}")

# --- 1台のカメラを処理するスレッド用の関数 (変更なし) ---
def process_camera_task(camera_id, yolo_model, yolo_lock, stop_event, db_client):
    """
    1台のカメラのキャプチャ、推論、アップロードを継続的に行う関数
    """
    
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"[Cam {camera_id}] Error: Could not open video device {camera_id}.")
        return

    print(f"[Cam {camera_id}] カメラ {camera_id} の処理を開始します。")
    
    try:
        while not stop_event.is_set():
            timestamp = datetime.datetime.now()
            
            # 1. 画像をキャプチャ
            ret, frame = cap.read()
            if not ret:
                print(f"[Cam {camera_id}] Error: Failed to capture image.")
                time.sleep(1) 
                continue
            
            # 2. AI推論 (★YOLOロックを使用★)
            log_time_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')
            print(f"[{log_time_str}] [Cam {camera_id}] [Inference] Running inference...")
            
            with yolo_lock:
                print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [Cam {camera_id}] YOLO Lock Acquired.")
                results = yolo_model(frame) 
                print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [Cam {camera_id}] YOLO Lock Released.")

            # 3. 検出結果の処理
            result = results[0] 
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            labels = result.boxes.cls.cpu().numpy()
            class_names = yolo_model.names 
            
            person_count = 0
            person_indices = []

            for i in range(len(boxes)):
                label_id = int(labels[i])
                score = scores[i]
                if score > 0.8 and label_id in class_names and class_names[label_id] == 'person':
                    person_count += 1
                    person_indices.append(i)

            print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [Cam {camera_id}] Detected {person_count} person(s).")

            # 4. 描画と送信 (人が1人以上いたら)
            if person_count > 0:
                for i in person_indices:
                    box = boxes[i].astype(int)
                    score = scores[i]
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                    label_text = f"person: {score:.2f}"
                    cv2.putText(frame, label_text, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                image_filename = f'capture_cam{camera_id}_{timestamp.strftime("%Y%m%d%H%M%S")}.jpg'
                image_path = f'./{image_filename}'
                cv2.imwrite(image_path, frame)
                print(f"[Cam {camera_id}] Saved annotated image: {image_path}")

                mac_dest_path = os.path.join(MAC_DEST_DIR, image_filename)
                send_file_to_mac(image_path, mac_dest_path)
                
                try:
                    os.remove(image_path)
                    print(f"[Cam {camera_id}] Deleted temporary image file: {image_filename}")
                except OSError as e:
                    print(f"[Cam {camera_id}] Error deleting file: {e}")
            
            # 5. Firebase Firestoreへのデータアップロード
            doc_data = {
                'count_time': timestamp,
                'count': person_count,
                'camera_id': camera_id 
            }
            
            collection_name = 'people_count' 
            try:
                db_client.collection(collection_name).add(doc_data)
                print(f"[Cam {camera_id}] Uploaded data to Firestore: {person_count} persons detected.")
            except Exception as e:
                print(f"[Cam {camera_id}] Error uploading data to Firestore: {e}")
            
            # 6. 処理間隔の待機
            stop_event.wait(10.0) 
    
    except Exception as e:
        print(f"[Cam {camera_id}] An error occurred in processing thread: {e}")
    finally:
        cap.release()
        print(f"[Cam {camera_id}] Camera {camera_id} released.")

# --- ★★★ メイン実行部分 (変更点) ★★★ ---
if __name__ == "__main__":
    
    # 1. Firebaseの初期化
    db = None
    try:
        service_account_path = 'serviceAccountKey.json'  # TODO(設定): serviceAccountKey.json
        cred = credentials.Certificate(service_account_path)
        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred)
        db = firestore.client()
        print("[Main] Firebase initialized successfully.")
    except Exception as e:
        print(f"[Main] Error initializing Firebase: {e}")
        exit()

    # 2. Macのフォルダ確認
    check_or_create_mac_folder()
    
    # 3. YOLOv9モデルを1回だけロードする
    print("[Main] Loading YOLOv9 model...")
    model = YOLO('yolov9c.pt')
    print("[Main] Model loaded successfully.")

    # 4. スレッド間で共有する「鍵」と「停止シグナル」を作成
    yolo_lock = threading.Lock() 
    stop_event = threading.Event()

    # 5. 監視スレッドを開始
    initial_time = time.time()
    monitor_thread = threading.Thread(target=monitor_system, args=(initial_time, stop_event))
    monitor_thread.daemon = True 
    monitor_thread.start()

    # 6. カメラ0処理スレッドを開始
    camera_0_thread = threading.Thread(target=process_camera_task, args=(0, model, yolo_lock, stop_event, db))
    camera_0_thread.daemon = True
    camera_0_thread.start()
    print("[Main] カメラ0スレッドを開始しました。")

    # --- ★★★ ここが変更点 ★★★ ---
    # 7. 5秒待機してからカメラ1スレッドを開始
    print("[Main] 5秒待機してからカメラ1スレッドを開始します...")
    try:
        time.sleep(5.0) # 5秒間待機
    except KeyboardInterrupt:
        print("\n[Main] 待機中に停止リクエストを受け取りました。")
        stop_event.set()
    
    if not stop_event.is_set():
        camera_1_thread = threading.Thread(target=process_camera_task, args=(2, model, yolo_lock, stop_event, db))
        camera_1_thread.daemon = True
        camera_1_thread.start()
        print("[Main] カメラ2スレッドを開始しました。")
    # --- ★★★ 変更点ここまで ★★★ ---

    # 8. メインスレッドは、停止シグナル (Ctrl+C) を待つだけ
    try:
        while not stop_event.is_set(): # 停止シグナルが来るまで待機
            time.sleep(1) 
    except KeyboardInterrupt:
        print("\n[Main] Program termination requested by user (Ctrl+C).")
    finally:
        # プログラム終了時に全スレッドを安全に停止
        print("[Main] Stopping all threads...")
        stop_event.set() # 全スレッドに停止シグナルを送信
        
        # 各スレッドが終了するのを待つ
        monitor_thread.join()
        camera_0_thread.join()
        if 'camera_1_thread' in locals() and camera_1_thread.is_alive():
            camera_1_thread.join()
        
        print("[Main] All threads stopped. System exiting.")
