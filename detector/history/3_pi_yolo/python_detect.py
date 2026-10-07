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
import shutil # ★ shutilライブラリをインポート

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
MAC_DIR_BEFORE = "/Users/<MAC_USER>/Desktop/raw_images"  # TODO(設定): /Users/<MAC_USER>/
MAC_DIR_AFTER = "/Users/<MAC_USER>/Desktop/received_images"  # TODO(設定): /Users/<MAC_USER>/


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

# --- ★★★ 1台のカメラを処理するスレッド用の関数 (保存処理を変更) ★★★ ---
def process_camera_task(camera_id, yolo_model, stop_event, db_client):
    """
    1台のカメラのキャプチャ、推論、アップロードを継続的に行う関数
    (YOLOロックを削除)
    """
    
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"[Cam {camera_id}] Error: Could not open video device {camera_id}.")
        return

    print(f"[Cam {camera_id}] カメラ {camera_id} の処理を開始します。")
    
    try:
        while not stop_event.is_set():
            timestamp = datetime.datetime.now()
            
            # --- ★★★ 抜けていた処理を追加 ★★★ ---
            # 1. 画像をキャプチャ
            ret, frame = cap.read()
            if not ret:
                print(f"[Cam {camera_id}] Error: Failed to capture image.")
                time.sleep(1) 
                continue
            
            # 2. AI推論
            log_time_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')
            print(f"[{log_time_str}] [Cam {camera_id}] [Inference] Running inference...")
            results = yolo_model(frame, verbose=False) # verbose=Falseでログを簡潔に
            # --- ★★★ 追加ここまで ★★★ ---

            # 3. 検出結果の処理
            result = results[0] 
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            labels = result.boxes.cls.cpu().numpy()
            class_names = yolo_model.names 
            
            person_count = 0
            person_indices = [] # ★ 描画するために復活

            for i in range(len(boxes)):
                label_id = int(labels[i])
                score = scores[i]
                # 信頼度の閾値を設定 (例: 0.3)
                if score > 0.3 and label_id in class_names and class_names[label_id] == 'person':
                    person_count += 1
                    person_indices.append(i) # ★ 描画するために復活

            print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [Cam {camera_id}] Detected {person_count} person(s).")

            # 4. 描画と送信 (人が1人以上いたら)
            if person_count > -1:
                # --- ★★★ 処理を「オリジナル」と「描画後」の2つに変更 ★★★ ---

                # --- 1. 撮影したままの写真 (Original) の処理 ---
                original_filename = f'original_cam{camera_id}_{timestamp.strftime("%Y%m%d%H%M%S")}.jpg'
                original_temp_path = f'./{original_filename}'
                
                # 撮影したままの 'frame' を保存
                cv2.imwrite(original_temp_path, frame)
                print(f"[Cam {camera_id}] Saved original image: {original_temp_path}")

                # Macに転送
                mac_dest_original = os.path.join(MAC_DIR_BEFORE, original_filename)
                send_file_to_mac(original_temp_path, mac_dest_original)
                try:
                    os.remove(original_temp_path)
                    print(f"[Cam {camera_id}] Deleted temporary original file: {original_temp_path}")
                except OSError as e:
                    print(f"[Cam {camera_id}] Error deleting original file: {e}")

                if person_count > 0:
                    # --- 2. バウンディングボックス付きの写真 (Annotated) の処理 ---
                    annotated_filename = f'annotated_cam{camera_id}_{timestamp.strftime("%Y%m%d%H%M%S")}.jpg'
                    annotated_temp_path = f'./{annotated_filename}'
                
                    # 元のフレームをコピーして、そちらに描画する
                    annotated_frame = frame.copy()
                
                    # ★ 描画処理を復活 ★
                    for i in person_indices:
                        box = boxes[i].astype(int)
                        score = scores[i]
                        cv2.rectangle(annotated_frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                        label_text = f"person: {score:.2f}"
                        cv2.putText(annotated_frame, label_text, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # 描画したフレームを保存
                    cv2.imwrite(annotated_temp_path, annotated_frame)
                    print(f"[Cam {camera_id}] Saved annotated image: {annotated_temp_path}")

                    # Macに転送
                    mac_dest_annotated = os.path.join(MAC_DIR_AFTER, annotated_filename)
                    send_file_to_mac(annotated_temp_path, mac_dest_annotated)     
                    try:
                        os.remove(annotated_temp_path)
                        print(f"[Cam {camera_id}] Deleted temporary annotated file: {annotated_temp_path}")
                    except OSError as e:
                        print(f"[Cam {camera_id}] Error deleting annotated file: {e}")      
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
            
            # 6. 処理間隔の待機 (例: 10秒)
            stop_event.wait(10.0) 
    
    except Exception as e:
        print(f"[Cam {camera_id}] An error occurred in processing thread: {e}") 
    
    finally:
        cap.release()
        print(f"[Cam {camera_id}] Camera {camera_id} released.")
# --- ★★★ メイン実行部分 (ローカルフォルダ作成を追加) ★★★ ---
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

    # 3. YOLOv12モデルを1回だけロードする
    print("[Main] Loading YOLOv12 model...")
    model = YOLO('yolo12n.pt') # または 'yolov8n.pt' など
    print("[Main] Model loaded successfully.")

    # 4. 「停止シグナル」を作成
    stop_event = threading.Event()

    # 5. 監視スレッドを開始
    initial_time = time.time()
    monitor_thread = threading.Thread(target=monitor_system, args=(initial_time, stop_event))
    monitor_thread.daemon = True 
    monitor_thread.start()

    # 6. カメラ0処理スレッドを開始 (ロック引数を削除)
    camera_0_thread = threading.Thread(target=process_camera_task, args=(0, model, stop_event, db))
    camera_0_thread.daemon = True
    camera_0_thread.start()
    print("[Main] カメラ0スレッドを開始しました。")

    # --- ★★★ カメラ1の処理を削除 ★★★ ---
    
    # 7. メインスレッドは、停止シグナル (Ctrl+C) を待つだけ
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
        
        print("[Main] All threads stopped. System exiting.")
