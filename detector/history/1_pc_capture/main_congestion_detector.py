#Firestoreなし
import os
import cv2
from ultralytics import YOLO
from datetime import datetime
import time
import mss
from PIL import Image
import numpy as np
import threading

RESET = "\033[0m"       # 全てのリセット
BOLD = "\033[1m"        # 太字
RED = "\033[91m"        # 赤
GREEN = "\033[92m"      # 緑
YELLOW = "\033[93m"     # 黄
BLUE = "\033[94m"       # 青
MAGENTA = "\033[95m"    # マゼンタ
CYAN = "\033[96m"       # シアン
WHITE = "\033[97m"      # 白 (明るい)

# --- 共通設定 ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# スクリーンショットとYOLO処理の両方に使用する画像フォルダ
INPUT_IMAGES_FOLDER = os.path.join(SCRIPT_DIR, 'bus_congestion_images')

# 検出結果の画像を保存するフォルダ (オプション)
OUTPUT_PROCESSED_FOLDER = os.path.join(SCRIPT_DIR, 'bus_congestion_processed_images')

# --- capture_screen.py の設定 ---
# QuickTime Playerウィンドウのキャプチャ領域の座標とサイズ
MONITOR_REGION = {
    "top": 100,  # 画面上端からのY座標 (ピクセル)
    "left": 50,  # 画面左端からのX座標 (ピクセル)
    "width": 800, # キャプチャする幅 (ピクセル)
    "height": 600 # キャプチャする高さ (ピクセル)
}
CAPTURE_INTERVAL_SCREEN = 5 # スクリーンショットを撮影する間隔（秒）

# --- process_images_with_yolo.py の設定 ---
YOLO_MODEL_PATH = 'yolov8n.pt' 
CONFIDENCE_THRESHOLD = 0.5 
TARGET_CLASSES = [0] # 'person' (人物) のみ検出対象とする
PROCESS_INTERVAL_YOLO = 5 # 新しい画像を監視する間隔（秒）

# --- フォルダの準備 ---
if not os.path.exists(INPUT_IMAGES_FOLDER):
    os.makedirs(INPUT_IMAGES_FOLDER)
    print(f"入力画像フォルダを作成しました: {INPUT_IMAGES_FOLDER}")

if not os.path.exists(OUTPUT_PROCESSED_FOLDER):
    os.makedirs(OUTPUT_PROCESSED_FOLDER)
    print(f"処理済み画像フォルダを作成しました: {OUTPUT_PROCESSED_FOLDER}")

# --- capture_screen.py の機能 ---
def capture_and_save_screenshot_task():
    """
    指定された領域のスクリーンショットを定期的に撮影し、ファイルに保存します。
    """
    with mss.mss() as sct:
        print(f"[CAPTURE] スクリーンショットの定期撮影を開始します。")
        print(f"[CAPTURE] キャプチャ領域: Left={MONITOR_REGION['left']}, Top={MONITOR_REGION['top']}, Width={MONITOR_REGION['width']}, Height={MONITOR_REGION['height']}")
        print(f"[CAPTURE] 保存先フォルダ: {INPUT_IMAGES_FOLDER}")
        print(f"[CAPTURE] 撮影間隔: {CAPTURE_INTERVAL_SCREEN}秒")
        
        try:
            while True:
                start_time = time.time()
                
                # スクリーンショットをキャプチャ
                sct_img = sct.grab(MONITOR_REGION)
                img = Image.frombytes("RGB", sct_img.size, sct_img.rgb)
                frame = np.array(img)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                # ファイル名にタイムスタンプを付加
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"bus_congestion_{timestamp}.jpg"
                filepath = os.path.join(INPUT_IMAGES_FOLDER, filename)

                # 画像を保存
                try:
                    cv2.imwrite(filepath, frame)
                    print(f"[CAPTURE] スクリーンショットを保存しました: {filepath}")
                except Exception as e:
                    print(f"[CAPTURE] スクリーンショットの保存中にエラーが発生しました: {e}")

                end_time = time.time()
                processing_time = end_time - start_time
                
                # 次のキャプチャまで待機
                if processing_time < CAPTURE_INTERVAL_SCREEN:
                    time.sleep(CAPTURE_INTERVAL_SCREEN - processing_time)
                else:
                    print(f"[CAPTURE] 警告: 処理時間が撮影間隔を超過しています ({processing_time:.2f}秒)。")

        except Exception as e:
            print(f"[CAPTURE] エラーによりキャプチャタスクが停止しました: {e}")

# --- process_images_with_yolo.py の機能 ---
def process_images_with_yolo_task():
    """
    指定フォルダ内の新しい画像をYOLOv8で処理し、結果を描画して保存します。
    """
    print(f"[YOLO] YOLOv8モデルをロード中: {YOLO_MODEL_PATH}...")
    model = YOLO(YOLO_MODEL_PATH)
    print("[YOLO] モデルのロードが完了しました。")

    if hasattr(model.device, 'type') and model.device.type == 'mps':
        print("[YOLO] GPU (MPS) が利用可能です。")
    elif hasattr(model.device, 'type') and model.device.type == 'cpu':
        print("[YOLO] CPUで実行します。")
    else:
        print(f"[YOLO] デバイスタイプを特定できませんでした: {model.device}")

    processed_files = set() # 既に処理したファイルを記録するセット

    try:
        while True:
            # 入力フォルダ内のすべての画像ファイルを取得
            all_files = [f for f in os.listdir(INPUT_IMAGES_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            # まだ処理されていない新しいファイルのみをフィルタリング
            new_files = [f for f in all_files if f not in processed_files]

            if not new_files:
                # print(f"[YOLO] 新しい画像が見つかりませんでした。{PROCESS_INTERVAL_YOLO}秒後に再確認します。") # ログが多すぎる場合コメントアウト
                time.sleep(PROCESS_INTERVAL_YOLO)
                continue

            print(f"[YOLO] 新しい画像 {len(new_files)} 件を処理します。")

            for filename in sorted(new_files): # 日時順に処理するためソート
                filepath = os.path.join(INPUT_IMAGES_FOLDER, filename)
                print(f"[YOLO]   処理中: {filename}")

                try:
                    img = cv2.imread(filepath)
                    if img is None:
                        print(f"[YOLO]   警告: 画像を読み込めませんでした: {filename}。スキップします。")
                        processed_files.add(filename) 
                        continue

                    results = model(img, conf=CONFIDENCE_THRESHOLD, classes=TARGET_CLASSES, verbose=False)

                    person_count = 0
                    annotated_img = img.copy() 
                    
                    for r in results:
                        for box in r.boxes:
                            cls_id = int(box.cls[0]) 
                            conf = float(box.conf[0]) 
                            
                            if cls_id == 0 and conf >= CONFIDENCE_THRESHOLD:
                                person_count += 1
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2) 
                                cv2.putText(annotated_img, f'Person {conf:.2f}', (x1, y1 - 10), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    print(f"{BLUE}[YOLO]     検出された人物の数: {person_count}{RESET}")
                    
                    # --- ここにDynamoDBへの書き込み処理を記述 ---
                    # 例:
                    # current_timestamp = datetime.now().isoformat()
                    # item = {
                    #     'BusStopId': 'BusStop_A', 
                    #     'Timestamp': current_timestamp,
                    #     'DetectedPeople': person_count,
                    #     'SourceImage': filename 
                    # }
                    # dynamo_table.put_item(Item=item)
                    # print(f"[YOLO]     DynamoDBにデータを書き込みました: {item}")
                    # --- ここまで ---

                    output_filepath = os.path.join(OUTPUT_PROCESSED_FOLDER, f"processed_{filename}")
                    cv2.imwrite(output_filepath, annotated_img)
                    print(f"[YOLO]     処理済み画像を保存しました: {output_filepath}")

                    # 処理が完了した元の画像ファイルを削除 (ストレージ容量の節約のため)
                    # 必要なければこの行をコメントアウトしてください
                    # os.remove(filepath)
                    # print(f"[YOLO]     元の画像を削除しました: {filepath}")

                    processed_files.add(filename)

                except Exception as e:
                    print(f"[YOLO]   エラー発生: {filename} の処理中にエラー: {e}。このファイルはスキップします。")
                    processed_files.add(filename) 
            
            time.sleep(PROCESS_INTERVAL_YOLO)

    except Exception as e:
        print(f"[YOLO] エラーによりYOLO処理タスクが停止しました: {e}")

# --- メイン実行部分 ---
if __name__ == "__main__":
    print("--- バス混雑検知システムを起動します ---")
    print("QuickTime PlayerでiPadの画面をミラーリングし、指定領域に配置してください。")
    print("スクリプトを開始するには Enter を押してください。終了するには Ctrl+C を押してください。")
    input() # ユーザーがEnterを押すまで待機

    # 各タスクを別々のスレッドで起動
    capture_thread = threading.Thread(target=capture_and_save_screenshot_task)
    yolo_process_thread = threading.Thread(target=process_images_with_yolo_task)

    # プログラム終了時にスレッドも終了するように設定
    capture_thread.daemon = True
    yolo_process_thread.daemon = True

    capture_thread.start()
    yolo_process_thread.start()

    try:
        # メインスレッドは、子スレッドが終了するまで待機（ここでは無限ループ）
        while True:
            time.sleep(1) 
    except KeyboardInterrupt:
        print("\n--- システムを停止します。しばらくお待ちください... ---")
        # デーモンスレッドはメインスレッド終了時に自動終了しますが、明示的なクリーンアップも考慮
    except Exception as e:
        print(f"メインスレッドで予期せぬエラーが発生しました: {e}")

    print("--- システムが正常に停止しました ---")
