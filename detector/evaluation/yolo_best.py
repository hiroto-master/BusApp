#Yolo処理のみ
import os
import cv2
from ultralytics import YOLO
from datetime import datetime # datetimeモジュールのインポート
import time                  # timeモジュールのインポート
import psutil # 新しく追加
import os     # psutil.Process(os.getpid()) のために念のため

# --- 設定 ---
# スクリーンショットが保存されているフォルダ
# 同じフォルダ構造を想定しています
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FOLDER = os.path.join("yolo_8",'input_images')

# 検出結果の画像を保存するフォルダ (オプション、不要ならコメントアウト可能)
OUTPUT_FOLDER_PROCESSED = os.path.join("yolo_8", 'output_images')

# AIモデルのパス（自動ダウンロードされます。ここでは'yolov8n.pt'を使用）
# 'yolov8n.pt' (nano) は軽量で高速、精度は標準
# 'yolov8s.pt' (small) は少し大きいが精度が良い
YOLO_MODEL_PATH = 'best.pt' 

# 検出の信頼度閾値（0.0〜1.0）
# 低くすると多くの物体を検出しますが、誤検出が増えます。高くすると確実な検出のみになります。
CONFIDENCE_THRESHOLD = 0.3

# 検出対象のクラスIDを指定（YOLOv8 COCOデータセットの場合）
# 0: person (人物)
# 全てのCOCOクラスは以下を参照: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco8.yaml
TARGET_CLASSES = [0] # 'person' (人物) のみ検出対象とする

# 新しい画像を監視する間隔（秒）
CAPTURE_INTERVAL = 5 # 例: 5秒ごとに新しい画像をチェック

# --- フォルダの準備 ---
# 処理済み画像を保存するフォルダが存在しない場合のみ作成します

def process_images_with_yolo():
    """
    指定フォルダ内の新しい画像をYOLOv8で処理し、結果を描画して保存します。
    """
    # このスレッドのプロセスオブジェクトを取得
    current_process = psutil.Process(os.getpid())

    print(f"YOLOモデルをロード中: {YOLO_MODEL_PATH}...")
    # モデルのロード。初回実行時に自動でダウンロードされます。
    model = YOLO(YOLO_MODEL_PATH)
    print("モデルのロードが完了しました。")

    # GPUが利用可能かチェック（MacのMシリーズチップはMPSが利用されます）
    if hasattr(model.device, 'type') and model.device.type == 'mps':
        print("GPU (MPS) が利用可能です。")
    elif hasattr(model.device, 'type') and model.device.type == 'cpu':
        print("CPUで実行します。")
    else:
        print(f"デバイスタイプを特定できませんでした: {model.device}")

    # 初期のメモリ使用量を記録 (YOLOモデルロード後)
    initial_memory_info = current_process.memory_info()
    initial_rss_mb = initial_memory_info.rss / (1024 * 1024) # Resident Set Size (RSS) MB
    print(f"YOLOモデルロード後の初期メモリ使用量 (RSS): {initial_rss_mb:.2f} MB")


    processed_files = set() # 既に処理したファイルを記録するセット

    try:
        while True:
            # 各処理サイクルの開始時のメモリ使用量を取得
            loop_start_memory_info = current_process.memory_info()
            loop_start_rss_mb = loop_start_memory_info.rss / (1024 * 1024)
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 処理サイクル開始時のメモリ使用量 (RSS): {loop_start_rss_mb:.2f} MB")

            # 入力フォルダ内のすべての画像ファイルを取得
            all_files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            # まだ処理されていない新しいファイルのみをフィルタリング
            new_files = [f for f in all_files if f not in processed_files]

            if not new_files:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] 新しい画像が見つかりませんでした。{CAPTURE_INTERVAL}秒後に再確認します。")
                time.sleep(CAPTURE_INTERVAL)
                continue

            print(f"[{datetime.now().strftime('%H:%M:%S')}] 新しい画像 {len(new_files)} 件を処理します。")

            # 新しいファイルを日付/時刻順に処理するためにソート
            for filename in sorted(new_files): 
                filepath = os.path.join(INPUT_FOLDER, filename)
                print(f"  処理中: {filename}")

                try:
                    # 画像の読み込み
                    img = cv2.imread(filepath)
                    if img is None:
                        print(f"  警告: 画像を読み込めませんでした: {filename}。スキップします。")
                        processed_files.add(filename) # 読み込めないファイルも処理済みとする
                        continue

                    # YOLOv8で推論を実行
                    # conf=CONFIDENCE_THRESHOLD: 信頼度閾値
                    # classes=TARGET_CLASSES: 検出対象のクラスID (0は人物)
                    # verbose=False: 詳細なログ出力を抑制
                    results = model(img, conf=CONFIDENCE_THRESHOLD, classes=TARGET_CLASSES, verbose=False)

                    # 検出結果を解析し、人数をカウント
                    person_count = 0
                    annotated_img = img.copy() # 結果描画用の画像のコピー
                    
                    for r in results:
                        # r.boxes は検出された物体（バウンディングボックス、信頼度、クラスID）を含む
                        for box in r.boxes:
                            cls_id = int(box.cls[0]) # クラスID
                            conf = float(box.conf[0]) # 信頼度
                            
                            # クラスIDが人物(0)で、信頼度閾値を超えている場合のみカウント
                            if cls_id == 0 and conf >= CONFIDENCE_THRESHOLD:
                                person_count += 1
                                # バウンディングボックスの描画（オプション）
                                # 緑色の四角とテキストを描画
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2) 
                                cv2.putText(annotated_img, f'Person {conf:.2f}', (x1, y1 - 10), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    print(f"    検出された人物の数: {person_count}")
                    

                    # 処理済み画像を保存 (オプション: 検出結果が描画された画像)
                    # この画像をS3にアップロードしてWebで表示することも考えられます
                    output_filepath = os.path.join(OUTPUT_FOLDER_PROCESSED, f"processed_{filename}")
                    cv2.imwrite(output_filepath, annotated_img)
                    print(f"    処理済み画像を保存しました: {output_filepath}")

                    # 処理が完了した元の画像ファイルを削除 (ストレージ容量の節約のため)
                    # 必要なければこの行をコメントアウトしてください
                    # os.remove(filepath)
                    # print(f"    元の画像を削除しました: {filepath}")

                    # 処理が完了したファイルを記録
                    #processed_files.add(filename)

                except Exception as e:
                    print(f"  エラー発生: {filename} の処理中にエラー: {e}")
                    # エラーが発生したファイルも処理済みとして、同じエラーでループしないようにする
                    processed_files.add(filename) 
            
            # すべての新しい画像を処理し終えたら、次のサイクルまで待機
            # 各処理サイクルの終了時のメモリ使用量を取得
            loop_end_memory_info = current_process.memory_info()
            loop_end_rss_mb = loop_end_memory_info.rss / (1024 * 1024)
            print(f"処理サイクル終了時のメモリ使用量 (RSS): {loop_end_rss_mb:.2f} MB")
            # ピークメモリ使用量も取得可能（ただし、Linux/macOSでのみ正確）
            if hasattr(loop_end_memory_info, 'peak_wset'):
                peak_wset_mb = loop_end_memory_info.peak_wset / (1024 * 1024)
                print(f"このプロセスのピークメモリ使用量 (peak_wset): {peak_wset_mb:.2f} MB (Linux/macOSのみ)")

            time.sleep(CAPTURE_INTERVAL)

    except KeyboardInterrupt:
        print("\n画像処理を停止します。")
    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}")

if __name__ == "__main__":
    # このスクリプトは、capture_screen.py と連携して使用することを想定しています。
    # まず capture_screen.py を実行して、画像が bus_congestion_images フォルダに生成されていることを確認してください。
    print("このスクリプトは、指定されたフォルダ内の新しい画像を監視し、YOLOv8で物体検出を行います。")
    print("スクリプトを開始するには Enter を押してください。停止するには Ctrl+C を押してください。")
    input() # ユーザーがEnterを押すまで待機

    process_images_with_yolo()
