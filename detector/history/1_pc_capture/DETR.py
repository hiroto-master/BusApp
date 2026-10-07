# DETR (MMDetection) による解析
import os
import cv2
from datetime import datetime
import time
import threading
import json 
import numpy as np
from PIL import Image

# Firebase Admin SDK のインポート
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

# --- ANSIカラーコード定義 ---
RESET = "\033[0m"       # 全てのリセット
BOLD = "\033[1m"        # 太字
RED = "\033[91m"        # 赤 (エラー、警告など)
GREEN = "\033[92m"      # 緑 (成功、重要な情報など)
YELLOW = "\033[93m"     # 黄 (警告、注意喚起など)
BLUE = "\033[94m"       # 青 (情報、一般的なメッセージなど)
MAGENTA = "\033[95m"    # マゼンタ
CYAN = "\033[96m"       # シアン (タスク名など)
WHITE = "\033[97m"      # 白 (明るい)


# --- 共通設定 ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# フォルダ名を 'DETR' に変更（お好みで調整してください）
INPUT_IMAGES_FOLDER = os.path.join("DETR", 'input_images')
OUTPUT_PROCESSED_FOLDER = os.path.join("DETR", 'output_images')

# --- DETR の設定 ---
# ★★★↓mim downloadで取得したファイル名に書き換えてください↓★★★
CONFIG_FILE = 'detr_r50_8xb2-150e_coco.py'
CHECKPOINT_FILE = 'detr_r50_8xb2-150e_coco_20221023_153551-436d03e8.pth'
# ★★★↑ mim downloadで取得したファイル名に書き換えてください↑★★★

CONFIDENCE_THRESHOLD = 0.5 
# MMDetectionのCOCOモデルでは 'person' のIDは 0 です
TARGET_CLASS_ID = 0 
PROCESS_INTERVAL_DETR = 5 # 新しい画像を監視する間隔（秒）


# --- Firebase 設定 ---
FIREBASE_SERVICE_ACCOUNT_KEY_PATH = os.path.join(SCRIPT_DIR, 'serviceAccountKey.json')  # TODO(設定): serviceAccountKey.json
FIRESTORE_COLLECTION_NAME = 'bus_congestion_data_detr' # (DETR用にコレクション名を変更)


# --- Firebase の初期化 ---
db = None # Firestoreクライアントをグローバルで保持
try:
    cred = credentials.Certificate(FIREBASE_SERVICE_ACCOUNT_KEY_PATH)
    # 既存のFirebaseアプリがないか確認してから初期化 (スレッディング時の重複初期化防止)
    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred)
    db = firestore.client()
    print(f"{GREEN}Firebase Admin SDKとFirestoreクライアントを初期化しました。{RESET}")
except Exception as e:
    print(f"{RED}Firebase Admin SDKの初期化に失敗しました: {e}{RESET}")
    print(f"{YELLOW}サービスアカウントキーのパスが正しいか確認してください。{RESET}")


# --- DETRによる画像処理タスク ---
def process_images_with_detr_task():
    """
    指定フォルダ内の新しい画像をDETR (MMDetection) で処理し、結果を描画して保存します。
    検出された人数をJSONデータとしてFirebase Cloud Firestoreにアップロードします。
    """
    # MMDetectionの初期化 (必須)
    register_all_modules()

    # デバイスの選択: MMDetectionは自動でGPUを検出しますが、明示的にCPUも指定可能
    # device = 'cuda:0' # GPUの場合
    device = 'cpu' # CPUの場合
    print(f"{CYAN}[DETR]{RESET} {BLUE}使用デバイス: {device}{RESET}")
    
    # DETRモデルのロード
    print(f"{CYAN}[DETR]{RESET} {BLUE}DETRモデルをロード中...{RESET}")
    try:
        model = init_detector(CONFIG_FILE, CHECKPOINT_FILE, device=device)
        print(f"{CYAN}[DETR]{RESET} {GREEN}モデルのロードが完了しました。{RESET}")
    except Exception as e:
        print(f"{CYAN}[DETR]{RED}モデルのロードに失敗しました: {e}{RESET}")
        print(f"{YELLOW}コンフィグ/チェックポイントのファイルパスが正しいか、MMDetectionの環境が正しいか確認してください。{RESET}")
        return # タスクを終了

    processed_files = set() # 既に処理したファイルを記録するセット

    try:
        while True:
            all_files = [f for f in os.listdir(INPUT_IMAGES_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            new_files = [f for f in all_files if f not in processed_files]

            if not new_files:
                time.sleep(PROCESS_INTERVAL_DETR)
                continue

            print(f"{CYAN}[DETR]{RESET} {GREEN}新しい画像 {len(new_files)} 件を処理します。{RESET}")

            for filename in sorted(new_files): # 日時順に処理するためソート
                filepath = os.path.join(INPUT_IMAGES_FOLDER, filename)
                print(f"{CYAN}[DETR]{RESET}   処理中: {filename}")

                try:
                    # 推論の実行 (MMDetectionはファイルパスを直接受け取れます)
                    result = inference_detector(model, filepath)

                    # --- 検出結果の解析 ---
                    person_count = 0
                    
                    # MMDetectionの結果オブジェクトからラベルとスコアを取得
                    labels = result.pred_instances.labels
                    scores = result.pred_instances.scores
                    
                    for i in range(len(labels)):
                        # 対象クラスが 'person' (ID=0) で、信頼度閾値以上の場合
                        if labels[i] == TARGET_CLASS_ID and scores[i] >= CONFIDENCE_THRESHOLD:
                            person_count += 1
                    
                    print(f"{CYAN}[DETR]{RESET}     {BOLD}検出された人物の数: {person_count}{RESET}")
                    
                    # --- Firebase Cloud Firestore へのデータアップロード処理 ---
                    if db: 
                        try:
                            data_to_firestore = {
                                'timestamp': datetime.now(), 
                                'detected_people_count': person_count,
                                'source_image_filename': filename,
                                'model_type': 'DETR' # どのモデルからのデータか明記
                            }
                            doc_ref = db.collection(FIRESTORE_COLLECTION_NAME).add(data_to_firestore)
                            print(f"{CYAN}[DETR]{RESET}     {GREEN}Firestoreにデータをアップロードしました。ドキュメントID: {doc_ref[1].id}{RESET}")
                        except Exception as fb_e:
                            print(f"{CYAN}[DETR]{RED}     Firestoreへのデータアップロード中にエラーが発生しました: {fb_e}{RESET}")
                    else:
                        print(f"{CYAN}[DETR]{RED}     Firebaseクライアントが初期化されていないため、アップロードはスキップされました。{RESET}")
                    # --- Firebase Cloud Firestore へのデータアップロード処理 終了 ---

                    # --- 処理済み画像の保存 ---
                    # MMDetectionの機能で結果を描画し、直接ファイルに保存
                    output_filepath = os.path.join(OUTPUT_PROCESSED_FOLDER, f"processed_{filename}")
                    
                    model.show_result(
                        filepath,
                        result,
                        score_thr=CONFIDENCE_THRESHOLD, # 閾値以上のものだけ描画
                        out_file=output_filepath
                    )
                    
                    print(f"{CYAN}[DETR]{RESET}     処理済み画像を保存しました: {output_filepath}")

                    # os.remove(filepath) # 必要であれば元画像を削除

                    processed_files.add(filename)

                except Exception as e:
                    print(f"{CYAN}[DETR]{RED}   エラー発生: {filename} の処理中にエラー: {e}。このファイルはスキップします。{RESET}")
                    processed_files.add(filename) 
            
            time.sleep(PROCESS_INTERVAL_DETR)

    except Exception as e:
        print(f"{CYAN}[DETR]{RED} エラーによりDETR処理タスクが停止しました: {e}{RESET}")

# --- メイン実行部分 ---
if __name__ == "__main__":
    print(f"{BOLD}--- バス混雑検知システム (DETR版) を起動します ---{RESET}")
    print(f"{BLUE}スクリプトを開始するには Enter を押してください。終了するには Ctrl+C を押してください。{RESET}")
    input() # ユーザーがEnterを押すまで待機

    # Faster R-CNNの代わりにDETRのタスクを実行
    detr_process_thread = threading.Thread(target=process_images_with_detr_task)
    detr_process_thread.daemon = True
    detr_process_thread.start()

    try:
        while True:
            time.sleep(1) 
    except KeyboardInterrupt:
        print(f"\n{BOLD}--- システムを停止します。しばらくお待ちください... ---{RESET}")
    except Exception as e:
        print(f"{RED}メインスレッドで予期せぬエラーが発生しました: {e}{RESET}")

    print(f"{BOLD}--- システムが正常に停止しました ---{RESET}")
