import os
import cv2
import supervision as sv # 描画のために維持
from ultralytics import YOLO
from datetime import datetime
import time
import psutil

# --- 設定 ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FOLDER = os.path.join(SCRIPT_DIR, "yolo_9", 'input_images')
OUTPUT_FOLDER_PROCESSED = os.path.join(SCRIPT_DIR, "yolo_9", 'output_images')

# 軽量モデル推奨 (リアルタイム性能重視)
YOLO_MODEL_PATH = 'yolo11n.pt' 

CONFIDENCE_THRESHOLD = 0.3
CAPTURE_INTERVAL = 1 
TARGET_CLASSES = [0] # 'person' (人物)

# --- フォルダ準備 ---
if not os.path.exists(OUTPUT_FOLDER_PROCESSED):
    os.makedirs(OUTPUT_FOLDER_PROCESSED)

def process_images_with_yolo():
    print(f"モデルをロード中: {YOLO_MODEL_PATH}...")
    model = YOLO(YOLO_MODEL_PATH)
    
    # --- Supervisionの初期化 (描画のみに使用) ---
    # 枠線を描くツール
    box_annotator = sv.BoxAnnotator(thickness=2)
    # ラベル(文字)を描くツール
    label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.5)
    
    current_process = psutil.Process(os.getpid())
    processed_files = set()

    try:
        while True:
            loop_start_rss_mb = current_process.memory_info().rss / (1024 * 1024)
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 処理サイクル開始時のメモリ使用量 (RSS): {loop_start_rss_mb:.2f} MB")

            all_files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            new_files = [f for f in all_files if f not in processed_files]

            if not new_files:
                time.sleep(CAPTURE_INTERVAL)
                continue

            print(f"[{datetime.now().strftime('%H:%M:%S')}] 新しい画像 {len(new_files)} 件を処理します。")

            for filename in sorted(new_files):
                filepath = os.path.join(INPUT_FOLDER, filename)
                print(f"  処理中: {filename}")

                try:
                    frame = cv2.imread(filepath)
                    if frame is None:
                        print(f"  画像読み込みエラー: {filename}")
                        processed_files.add(filename)
                        continue

                    # 1. 推論実行
                    # TARGET_CLASSES (0=person)を指定
                    results = model(frame, conf=CONFIDENCE_THRESHOLD, classes=TARGET_CLASSES, verbose=False, agnostic_nms=True)[0]

                    # 2. Supervision形式に変換し、検出された人物数を取得
                    detections = sv.Detections.from_ultralytics(results)
                    person_count = len(detections) # 画面全体の検出数をカウント
                    
                    print(f"    検出された人物の総数: {person_count}人")

                    # --- 描画処理 ---
                    # 1. 枠を描画
                    frame = box_annotator.annotate(scene=frame, detections=detections)
                    
                    # 2. ラベルを描画
                    frame = label_annotator.annotate(scene=frame, detections=detections)
                    
                    # 3. カウント数を画像左上に描画（任意）
                    cv2.putText(
                        frame,
                        f"Total Persons: {person_count}",
                        (10, 30), # 座標 (x, y)
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, # スケール
                        (0, 255, 0), # 色 (緑)
                        2 # 太さ
                    )

                    # 保存
                    output_filepath = os.path.join(OUTPUT_FOLDER_PROCESSED, f"processed_{filename}")
                    cv2.imwrite(output_filepath, frame)
                    print(f"    処理済み画像を保存しました: {output_filepath}")

                    processed_files.add(filename)

                except Exception as e:
                    print(f"  エラー発生: {filename} の処理中にエラー: {e}")
                    processed_files.add(filename)

            loop_end_rss_mb = current_process.memory_info().rss / (1024 * 1024)
            print(f"処理サイクル終了時のメモリ使用量 (RSS): {loop_end_rss_mb:.2f} MB")

            time.sleep(CAPTURE_INTERVAL)

    except KeyboardInterrupt:
        print("\n画像処理を停止します。")
    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}")

if __name__ == "__main__":
    print("このスクリプトは、フォルダ内の新しい画像を監視し、画像全体に映る人物をカウントします。")
    print("スクリプトを開始するには Enter を押してください。")
    input()

    process_images_with_yolo()
