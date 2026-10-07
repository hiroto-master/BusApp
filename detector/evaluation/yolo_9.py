# Yolo処理のみ (YOLOv9にアップグレード)
import os
import cv2
from ultralytics import YOLO
from datetime import datetime
import time
import psutil

# --- 設定 ---
# フォルダ設定は変更なし
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FOLDER = os.path.join(SCRIPT_DIR, 'bus_congestion_images')
OUTPUT_FOLDER_PROCESSED = os.path.join(SCRIPT_DIR, 'bus_congestion_processed_images')

# ★変更点1: AIモデルをYOLOv9に更新
# 'yolov9c.pt' (YOLOv9 Compact): 高精度と速度のバランスが良いモデル
# 'yolov8s.pt' (YOLOv8 Small): YOLOv8系で精度を上げたい場合
YOLO_MODEL_PATH = 'yolov9c.pt'

# その他の設定は変更なし
CONFIDENCE_THRESHOLD = 0.5
TARGET_CLASSES = [0]  # 'person' (人物)
CAPTURE_INTERVAL = 5

# --- フォルダの準備 ---
if not os.path.exists(OUTPUT_FOLDER_PROCESSED):
    os.makedirs(OUTPUT_FOLDER_PROCESSED)
    print(f"出力フォルダを作成しました: {OUTPUT_FOLDER_PROCESSED}")

def process_images_with_yolo():
    """
    指定フォルダ内の新しい画像をYOLOv9で処理し、結果を描画して保存します。
    """
    current_process = psutil.Process(os.getpid())

    print(f"YOLOv9モデルをロード中: {YOLO_MODEL_PATH}...")
    # 指定したモデルがなければ、自動的にダウンロードされます
    model = YOLO(YOLO_MODEL_PATH)
    print("モデルのロードが完了しました。")

    if hasattr(model.device, 'type') and model.device.type == 'mps':
        print("GPU (MPS) が利用可能です。")
    else:
        print("CPUで実行します。")

    initial_memory_info = current_process.memory_info()
    initial_rss_mb = initial_memory_info.rss / (1024 * 1024)
    print(f"YOLOモデルロード後の初期メモリ使用量 (RSS): {initial_rss_mb:.2f} MB")

    processed_files = set()

    try:
        while True:
            loop_start_memory_info = current_process.memory_info()
            loop_start_rss_mb = loop_start_memory_info.rss / (1024 * 1024)
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 処理サイクル開始時のメモリ使用量 (RSS): {loop_start_rss_mb:.2f} MB")

            all_files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            new_files = [f for f in all_files if f not in processed_files]

            if not new_files:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] 新しい画像が見つかりませんでした。{CAPTURE_INTERVAL}秒後に再確認します。")
                time.sleep(CAPTURE_INTERVAL)
                continue

            print(f"[{datetime.now().strftime('%H:%M:%S')}] 新しい画像 {len(new_files)} 件を処理します。")

            for filename in sorted(new_files):
                filepath = os.path.join(INPUT_FOLDER, filename)
                print(f"  処理中: {filename}")

                try:
                    # YOLOv9で推論を実行 (画像パスを直接渡すのが効率的)
                    results = model(filepath, conf=CONFIDENCE_THRESHOLD, classes=TARGET_CLASSES, verbose=False)
                    
                    # resultsはリストなので、最初の要素を取得
                    result = results[0]

                    # ★変更点2: 結果の解析をよりシンプルに
                    # 'person'クラス(ID:0)のボックスのみをフィルタリング
                    person_boxes = [box for box in result.boxes if box.cls == 0]
                    person_count = len(person_boxes)
                    
                    print(f"    検出された人物の数: {person_count}")

                    # --- DynamoDBへの書き込み処理などはここに記述 ---

                    # ★変更点3: 結果の描画をライブラリの機能に任せる
                    # result.plot() は検出結果を元画像に描画したNumPy配列を返す
                    annotated_image = result.plot()
                    
                    output_filepath = os.path.join(OUTPUT_FOLDER_PROCESSED, f"processed_{filename}")
                    cv2.imwrite(output_filepath, annotated_image)
                    print(f"    処理済み画像を保存しました: {output_filepath}")

                    # 元の画像を削除するオプションは維持
                    # os.remove(filepath)
                    # print(f"    元の画像を削除しました: {filepath}")

                    processed_files.add(filename)

                except Exception as e:
                    print(f"  エラー発生: {filename} の処理中にエラー: {e}")
                    processed_files.add(filename)

            loop_end_memory_info = current_process.memory_info()
            loop_end_rss_mb = loop_end_memory_info.rss / (1024 * 1024)
            print(f"処理サイクル終了時のメモリ使用量 (RSS): {loop_end_rss_mb:.2f} MB")

            time.sleep(CAPTURE_INTERVAL)

    except KeyboardInterrupt:
        print("\n画像処理を停止します。")
    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}")

if __name__ == "__main__":
    print("このスクリプトは、指定されたフォルダ内の新しい画像を監視し、YOLOv9で物体検出を行います。")
    print("スクリプトを開始するには Enter を押してください。停止するには Ctrl+C を押してください。")
    input()

    process_images_with_yolo()
