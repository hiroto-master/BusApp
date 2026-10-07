# Cascade R-CNNによる画像処理
import os
import cv2
import time
import psutil
from datetime import datetime
import asyncio # MMDetectionで必要
from mmdet.apis import init_detector, inference_detector
from mmdet.utils import register_all_modules

# --- 設定 ---
# フォルダ設定
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FOLDER = os.path.join("Cascade-R-CNN", 'input_images')
OUTPUT_FOLDER_PROCESSED = os.path.join("Cascade-R-CNN", 'output_images')

# ★変更点1: AIモデルをCascade R-CNNに指定
CONFIG_FILE = 'cascade_rcnn_r50_fpn_1x_coco.py'
CHECKPOINT_FILE = 'cascade_rcnn_r50_fpn_1x_coco_20200316-3dc56deb.pth' # mim downloadで取得したファイル名

# その他の設定
CONFIDENCE_THRESHOLD = 0.5
TARGET_CLASS_ID = 0  # COCOデータセットの 'person' (人物) はID 0で共通
CAPTURE_INTERVAL = 5

async def process_images_with_cascade():
    """
    指定フォルダ内の新しい画像をCascade R-CNNで処理し、結果を描画して保存します。
    """
    current_process = psutil.Process(os.getpid())
    
    # MMDetectionの初期化
    register_all_modules()

    print(f"Cascade R-CNNモデルをロード中...")
    # モデルのロード。GPUがない場合は device='cpu' を指定
    model = init_detector(CONFIG_FILE, CHECKPOINT_FILE, device='cpu')
    print("モデルのロードが完了しました。")

    initial_memory_info = current_process.memory_info()
    initial_rss_mb = initial_memory_info.rss / (1024 * 1024)
    print(f"モデルロード後の初期メモリ使用量 (RSS): {initial_rss_mb:.2f} MB")

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
                    # MMDetectionで推論を実行
                    result = inference_detector(model, filepath)

                    # ★変更点2: 結果の解析を手動で行う
                    person_count = 0
                    # 検出結果からラベルとスコア（信頼度）を取得
                    labels = result.pred_instances.labels
                    scores = result.pred_instances.scores
                    
                    for i in range(len(labels)):
                        if labels[i] == TARGET_CLASS_ID and scores[i] >= CONFIDENCE_THRESHOLD:
                            person_count += 1
                            
                    print(f"    検出された人物の数: {person_count}")

                    # --- DynamoDBへの書き込み処理などはここに記述 ---

                    # ★変更点3: 結果の描画をMMDetectionの機能で行い、直接ファイルに保存
                    output_filepath = os.path.join(OUTPUT_FOLDER_PROCESSED, f"processed_{filename}")
                    
                    # model.show_result() は結果を描画してファイルに保存する機能を持つ
                    model.show_result(
                        filepath,
                        result,
                        score_thr=CONFIDENCE_THRESHOLD, # 閾値以上のものだけ描画
                        out_file=output_filepath
                    )
                    
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
    print("このスクリプトは、指定されたフォルダ内の新しい画像を監視し、Cascade R-CNNで物体検出を行います。")
    print("スクリプトを開始するには Enter を押してください。停止するには Ctrl+C を押してください。")
    input()

    asyncio.run(process_images_with_cascade())
