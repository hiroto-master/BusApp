import cv2
import threading
import time
import datetime
import torch
import torchvision
from torchvision.transforms import functional as F
from ultralytics import YOLO

# --- モデル設定 ---
# (注意: 'yolov12m.pt' は存在しないため、'yolov8m.pt' や 'yolov9c.pt' など実在のモデルを使います)
YOLO_MODEL_PATH = 'yolo12m.pt' 
CAMERA_ID_YOLO = 1       # YOLOが担当するカメラID
CAMERA_ID_FRCNN = 0      # Faster R-CNNが担当するカメラID

# クラスID 'person' (COCOデータセットでは 1)
PERSON_CLASS_ID_FRCNN = 1
# YOLO (ultralytics) では 'person' はクラスID 0
PERSON_CLASS_ID_YOLO = 0

# --- Faster R-CNN のロードと設定 ---
def load_faster_rcnn_model():
    """
    torchvisionからFaster R-CNN (ResNet-50) モデルをロードし、評価モードにする。
    """
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] [Main] Loading Faster R-CNN model...")
    # pretrained=True で学習済みモデルをダウンロード
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    # .eval() で推論（評価）モードに切り替え
    model.eval()
    
    # ラズパイなどCPUで動かす場合は 'cpu' を指定
    # (GPUがある場合は 'cuda' にすると高速)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] [Main] Faster R-CNN model loaded on {device}.")
    return model, device

# --- YOLO のロード ---
def load_yolo_model():
    """
    UltralyticsからYOLOモデルをロードする。
    """
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] [Main] Loading YOLO model ({YOLO_MODEL_PATH})...")
    
    # ラズパイなどCPUで動かす場合は device='cpu' を指定
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(YOLO_MODEL_PATH)
    model.to(device) # 明示的にデバイスを指定
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] [Main] YOLO model loaded on {device}.")
    return model, device

# --- 各カメラを処理するスレッド関数 ---
def process_camera_task(camera_id, model_name, model, device, stop_event):
    """
    1台のカメラのキャプチャと推論をループ実行する関数（スレッドターゲット）
    """
    
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"[Cam {camera_id}][{model_name}] Error: Could not open video device {camera_id}.")
        return

    print(f"[Cam {camera_id}][{model_name}] Processing thread started.")

    try:
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                print(f"[Cam {camera_id}][{model_name}] Error: Failed to capture frame.")
                stop_event.wait(1.0) # 1秒待機して再試行
                continue

            person_count = 0
            start_time = time.time()

            # --- モデルに応じて推論方法を分岐 ---
            
            if model_name == "YOLO":
                # YOLO (Ultralytics) での推論
                results = model(frame, verbose=False, device=device)
                
                for r in results:
                    # 検出結果から 'person' (ID 0) だけを数える
                    for cls in r.boxes.cls:
                        if int(cls) == PERSON_CLASS_ID_YOLO:
                            person_count += 1
            
            elif model_name == "Faster R-CNN":
                # Faster R-CNN (Torchvision) での推論
                
                # 1. OpenCV画像(BGR)をRGBに変換
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # 2. PyTorchテンソルに変換 (H, W, C) -> (C, H, W)
                tensor_frame = F.to_tensor(rgb_frame)
                # 3. デバイス(CPU/GPU)に送り、バッチ次元を追加 [C, H, W] -> [1, C, H, W]
                tensor_frame = tensor_frame.to(device).unsqueeze(0)
                
                # 4. 推論実行 (勾配計算をオフにして高速化)
                with torch.no_grad():
                    outputs = model(tensor_frame)
                
                # 5. 結果の処理
                output = outputs[0] # バッチの最初の結果を取得
                for i in range(len(output['labels'])):
                    label = output['labels'][i].item()
                    score = output['scores'][i].item()
                    
                    # 信頼度0.5以上 かつ 'person' (ID 1) の場合
                    if score > 0.5 and label == PERSON_CLASS_ID_FRCNN:
                        person_count += 1

            # --- 推論ここまで ---
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            log_time = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]
            print(f"[{log_time}] [Cam {camera_id}][{model_name: <12}] Detected: {person_count: >2} persons (Proc. Time: {processing_time:.3f} s)")
            
            # CPUを占有しすぎないよう、短時間待機
            # (ただし、Faster R-CNNが重すぎる場合、この待機はほぼ意味をなさない)
            stop_event.wait(0.01) # 0.01秒待機

    except Exception as e:
        print(f"[Cam {camera_id}][{model_name}] An error occurred: {e}")
    finally:
        cap.release()
        print(f"[Cam {camera_id}][{model_name}] Processing thread stopped and camera released.")

# --- メイン実行部分 ---
if __name__ == "__main__":
    
    # 1. 両方のモデルをメインスレッドでロードする (起動時に1回だけ)
    try:
        yolo_model, yolo_device = load_yolo_model()
        frcnn_model, frcnn_device = load_faster_rcnn_model()
    except Exception as e:
        print(f"[Main] Error loading models: {e}")
        print("[Main] torch または ultralytics が正しくインストールされているか確認してください。")
        exit()

    # 2. 全スレッドを安全に停止させるための「停止シグナル」を作成
    stop_event = threading.Event()

    # 3. Faster R-CNN用スレッドを作成
    thread_frcnn = threading.Thread(
        target=process_camera_task, 
        args=(CAMERA_ID_FRCNN, "Faster R-CNN", frcnn_model, frcnn_device, stop_event),
        daemon=True # メインスレッドが終了したら道連れで終了する
    )

    # 4. YOLO用スレッドを作成
    thread_yolo = threading.Thread(
        target=process_camera_task, 
        args=(CAMERA_ID_YOLO, "YOLO", yolo_model, yolo_device, stop_event),
        daemon=True
    )

    # 5. 両方のスレッドを同時にスタート！
    thread_frcnn.start()
    thread_yolo.start()
    
    print(f"\n[{datetime.datetime.now().strftime('%H:%M:%S')}] [Main] Both camera processing threads started.")
    print("[Main] Press Ctrl+C to stop the program.")

    # 6. メインスレッドは停止シグナル (Ctrl+C) を待つだけ
    try:
        # メインスレッドが生きている限り、デーモンスレッドも動き続ける
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[Main] Program termination requested by user (Ctrl+C).")
    finally:
        # プログラム終了時に全スレッドを安全に停止
        print("[Main] Stopping all threads...")
        stop_event.set() # 全スレッドに停止シグナルを送信
        
        # 各スレッドが終了するのを待つ (join)
        thread_frcnn.join(timeout=5.0) # 最大5秒待つ
        thread_yolo.join(timeout=5.0)
        
        print("[Main] All threads stopped. System exiting.")
