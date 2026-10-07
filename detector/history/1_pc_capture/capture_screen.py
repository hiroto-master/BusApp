#画面キャプチャのみ
import mss
import numpy as np
import cv2
from PIL import Image
import time
import os
from datetime import datetime

# --- 設定 ---
# QuickTime Playerウィンドウのキャプチャ領域の座標とサイズを設定
# あなたの環境に合わせて調整してください
# 例: "top": 100 は画面上端から100ピクセル下、"left": 50 は画面左端から50ピクセル右
# "width": 800 はキャプチャする幅800ピクセル、"height": 600 はキャプチャする高さ600ピクセル
MONITOR_REGION = {
    "top": 100,  # 画面上端からのY座標 (ピクセル)
    "left": 50,  # 画面左端からのX座標 (ピクセル)
    "width": 800, # キャプチャする幅 (ピクセル)
    "height": 600 # キャプチャする高さ (ピクセル)
}

# スクリーンショットの保存先フォルダ
# 現在のスクリプトがあるディレクトリ（つまりプロジェクトフォルダ）の直下に
# 'bus_congestion_images'というフォルダを作成します。
# 例: ~/Desktop/bus_congestion_project/bus_congestion_images
# フォルダ名は分かりやすいように 'bus_congestion_images' としました。
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FOLDER = os.path.join(SCRIPT_DIR, 'bus_congestion_images') 

# スクリーンショットを撮影する間隔（秒）
CAPTURE_INTERVAL = 3 # 例: 3秒ごとに撮影

# --- フォルダの準備 ---
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)
    print(f"出力フォルダを作成しました: {OUTPUT_FOLDER}")

def capture_and_save_screenshot():
    """
    指定された領域のスクリーンショットを撮影し、ファイルに保存します。
    """
    with mss.mss() as sct:
        # 指定領域のスクリーンショットをキャプチャ
        sct_img = sct.grab(MONITOR_REGION)
        
        # PIL Imageオブジェクトに変換
        img = Image.frombytes("RGB", sct_img.size, sct_img.rgb)
        
        # NumPy配列に変換 (OpenCVで保存するため)
        frame = np.array(img)
        
        # PILのRGBからOpenCVのBGR形式に変換
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # ファイル名にタイムスタンプを付加
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"bus_congestion_{timestamp}.jpg"
        filepath = os.path.join(OUTPUT_FOLDER, filename)

        # 画像を保存
        try:
            cv2.imwrite(filepath, frame)
            print(f"スクリーンショットを保存しました: {filepath}")
            return True
        except Exception as e:
            print(f"スクリーンショットの保存中にエラーが発生しました: {e}")
            return False

def main():
    print("スクリーンショットの定期撮影を開始します。")
    print(f"キャプチャ領域: Left={MONITOR_REGION['left']}, Top={MONITOR_REGION['top']}, Width={MONITOR_REGION['width']}, Height={MONITOR_REGION['height']}")
    print(f"保存先フォルダ: {OUTPUT_FOLDER}")
    print(f"撮影間隔: {CAPTURE_INTERVAL}秒")
    print("停止するには Ctrl+C を押してください。")

    try:
        while True:
            start_time = time.time()
            
            # スクリーンショットを撮影して保存
            capture_and_save_screenshot()

            end_time = time.time()
            processing_time = end_time - start_time
            
            # 次のキャプチャまで待機
            if processing_time < CAPTURE_INTERVAL:
                time.sleep(CAPTURE_INTERVAL - processing_time)
            else:
                print(f"警告: 処理時間が撮影間隔を超過しています ({processing_time:.2f}秒)。")

    except KeyboardInterrupt:
        print("\nスクリーンショットの撮影を停止しました。")
    except Exception as e:
        print(f"エラーが発生しました: {e}")

if __name__ == "__main__":
    main()
