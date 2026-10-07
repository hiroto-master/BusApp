#キャプチャしてからfaster解析
import os
import cv2
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from datetime import datetime
import time
import mss
from PIL import Image
import numpy as np
import threading
import json 

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
INPUT_IMAGES_FOLDER = os.path.join(SCRIPT_DIR, 'bus_congestion_images')
OUTPUT_PROCESSED_FOLDER = os.path.join(SCRIPT_DIR, 'bus_congestion_processed_images')

# --- capture_screen.py の設定 ---
MONITOR_REGION = {
    "top": 100,  # 画面上端からのY座標 (ピクセル)
    "left": 50,  # 画面左端からのX座標 (ピクセル)
    "width": 800, # キャプチャする幅 (ピクセル)
    "height": 600 # キャプチャする高さ (ピクセル)
}
CAPTURE_INTERVAL_SCREEN = 5 # スクリーンショットを撮影する間隔（秒）

# --- process_images_with_fasterrcnn.py の設定 ---
CONFIDENCE_THRESHOLD = 0.5 
TARGET_CLASSES = [1] # COCOデータセットにおける 'person' のIDは 1 です (YOLOv8の 0 とは異なります)
PROCESS_INTERVAL_FASTER_RCNN = 5 # 新しい画像を監視する間隔（秒）

# COCOデータセットのクラス名リスト (必要に応じて確認用)
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'deer', 'dog', 'horse', 'cat', 'elephant', 'bear',
    'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase',
    'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
    'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
    'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
    'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet',
    'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


# --- Firebase 設定 ---
# ダウンロードしたサービスアカウントキーのパスを指定
# (例: serviceAccountKey.json)
FIREBASE_SERVICE_ACCOUNT_KEY_PATH = os.path.join(SCRIPT_DIR, 'serviceAccountKey.json')  # TODO(設定): serviceAccountKey.json
# ↑↑↑ ここをダウンロードしたJSONファイルのパスに書き換えてください ↑↑↑

# Firestoreのコレクション名（データの種類を表すフォルダのようなもの）
FIRESTORE_COLLECTION_NAME = 'bus_congestion_data' 

# --- フォルダの準備 ---
if not os.path.exists(INPUT_IMAGES_FOLDER):
    os.makedirs(INPUT_IMAGES_FOLDER)
    print(f"{GREEN}入力画像フォルダを作成しました: {INPUT_IMAGES_FOLDER}{RESET}")

if not os.path.exists(OUTPUT_PROCESSED_FOLDER):
    os.makedirs(OUTPUT_PROCESSED_FOLDER)
    print(f"{GREEN}処理済み画像フォルダを作成しました: {OUTPUT_PROCESSED_FOLDER}{RESET}")

# --- Firebase の初期化 ---
db = None # Firestoreクライアントをグローバルで保持
try:
    cred = credentials.Certificate(FIREBASE_SERVICE_ACCOUNT_KEY_PATH)
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    print(f"{GREEN}Firebase Admin SDKとFirestoreクライアントを初期化しました。{RESET}")
except Exception as e:
    print(f"{RED}Firebase Admin SDKの初期化に失敗しました: {e}{RESET}")
    print(f"{YELLOW}サービスアカウントキーのパスが正しいか、ダウンロードされているか確認してください。{RESET}")


# --- capture_screen.py の機能 ---
def capture_and_save_screenshot_task():
    """
    指定された領域のスクリーンショットを定期的に撮影し、ファイルに保存します。
    """
    with mss.mss() as sct:
        print(f"{CYAN}[CAPTURE]{RESET} {BLUE}スクリーンショットの定期撮影を開始します。{RESET}")
        print(f"{CYAN}[CAPTURE]{RESET} {BLUE}キャプチャ領域: Left={MONITOR_REGION['left']}, Top={MONITOR_REGION['top']}, Width={MONITOR_REGION['width']}, Height={MONITOR_REGION['height']}{RESET}")
        print(f"{CYAN}[CAPTURE]{RESET} {BLUE}保存先フォルダ: {INPUT_IMAGES_FOLDER}{RESET}")
        print(f"{CYAN}[CAPTURE]{RESET} {BLUE}撮影間隔: {CAPTURE_INTERVAL_SCREEN}秒{RESET}")
        
        try:
            while True:
                start_time = time.time()
                
                sct_img = sct.grab(MONITOR_REGION)
                img = Image.frombytes("RGB", sct_img.size, sct_img.rgb)
                frame = np.array(img)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"bus_congestion_{timestamp}.jpg"
                filepath = os.path.join(INPUT_IMAGES_FOLDER, filename)

                try:
                    cv2.imwrite(filepath, frame)
                    print(f"{CYAN}[CAPTURE]{RESET} スクリーンショットを保存しました: {filepath}")
                except Exception as e:
                    print(f"{CYAN}[CAPTURE]{RESET} {RED}スクリーンショットの保存中にエラーが発生しました: {e}{RESET}")

                end_time = time.time()
                processing_time = end_time - start_time
                
                if processing_time < CAPTURE_INTERVAL_SCREEN:
                    time.sleep(CAPTURE_INTERVAL_SCREEN - processing_time)
                else:
                    print(f"{CYAN}[CAPTURE]{RESET} {YELLOW}警告: 処理時間が撮影間隔を超過しています ({processing_time:.2f}秒)。{RESET}")

        except Exception as e:
            print(f"{CYAN}[CAPTURE]{RED} エラーによりキャプチャタスクが停止しました: {e}{RESET}")

# --- process_images_with_fasterrcnn.py の機能 ---
def process_images_with_fasterrcnn_task():
    """
    指定フォルダ内の新しい画像をFaster R-CNNで処理し、結果を描画して保存します。
    検出された人数をJSONデータとしてFirebase Cloud Firestoreにアップロードします。
    """
    # デバイスの選択: GPUが利用可能ならGPU (CUDA)、なければCPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"{CYAN}[Faster R-CNN]{RESET} {BLUE}使用デバイス: {device}{RESET}")
    
    # Faster R-CNNモデルのロード (学習済みモデル)
    # COCOデータセットで学習済みのresnet50_fpnバックボーンを持つFaster R-CNNモデルを使用
    print(f"{CYAN}[Faster R-CNN]{RESET} {BLUE}Faster R-CNNモデルをロード中...{RESET}")
    model = fasterrcnn_resnet50_fpn(weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    model.to(device) # モデルをデバイスに移動
    model.eval() # 評価モードに設定 (ドロップアウトやバッチ正規化が推論モードになる)
    print(f"{CYAN}[Faster R-CNN]{RESET} {GREEN}モデルのロードが完了しました。{RESET}")

    processed_files = set() # 既に処理したファイルを記録するセット

    try:
        while True:
            all_files = [f for f in os.listdir(INPUT_IMAGES_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            new_files = [f for f in all_files if f not in processed_files]

            if not new_files:
                time.sleep(PROCESS_INTERVAL_FASTER_RCNN)
                continue

            print(f"{CYAN}[Faster R-CNN]{RESET} {GREEN}新しい画像 {len(new_files)} 件を処理します。{RESET}")

            for filename in sorted(new_files): # 日時順に処理するためソート
                filepath = os.path.join(INPUT_IMAGES_FOLDER, filename)
                print(f"{CYAN}[Faster R-CNN]{RESET}   処理中: {filename}")

                try:
                    img_np = cv2.imread(filepath)
                    if img_np is None:
                        print(f"{CYAN}[Faster R-CNN]{RESET} {YELLOW}警告: 画像を読み込めませんでした: {filename}。スキップします。{RESET}")
                        processed_files.add(filename) 
                        continue
                    
                    # OpenCV (BGR) から PIL Image (RGB) に変換し、PyTorchテンソルへ
                    img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
                    img_pil = Image.fromarray(img_rgb)
                    
                    # 画像をPyTorchテンソルに変換 (正規化も含まれる)
                    # モデルはバッチ形式 [B, C, H, W] を期待するので unsqueeze(0) でバッチ次元を追加
                    img_tensor = F.to_tensor(img_pil).to(device)

                    # 推論の実行
                    # モデルはリスト形式のテンソルを入力として受け取る
                    with torch.no_grad(): # 勾配計算を無効化 (推論時は不要でメモリ・速度向上)
                        prediction = model([img_tensor])

                    person_count = 0
                    annotated_img = img_np.copy() # 描画用に元の画像をコピー
                    
                    # 検出結果の解析
                    # prediction[0] は1つの画像に対する検出結果の辞書
                    boxes = prediction[0]['boxes'].cpu().numpy()
                    labels = prediction[0]['labels'].cpu().numpy()
                    scores = prediction[0]['scores'].cpu().numpy()

                    for i in range(len(boxes)):
                        score = scores[i]
                        label = labels[i]
                        
                        # 対象クラスが 'person' (ID=1) で、信頼度閾値以上の場合
                        if label == TARGET_CLASSES[0] and score >= CONFIDENCE_THRESHOLD:
                            person_count += 1
                            x1, y1, x2, y2 = map(int, boxes[i])
                            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2) 
                            cv2.putText(annotated_img, f'Person {score:.2f}', (x1, y1 - 10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    print(f"{CYAN}[Faster R-CNN]{RESET}     {BOLD}検出された人物の数: {person_count}{RESET}")
                    
                    # --- Firebase Cloud Firestore へのデータアップロード処理 ---
                    if db: # Firestoreクライアントが正常に初期化されている場合のみ実行
                        try:
                            # Firestoreに保存するデータ
                            data_to_firestore = {
                                'timestamp': datetime.now(), # Firestoreはdatetimeオブジェクトを直接扱える
                                'detected_people_count': person_count,
                                'source_image_filename': filename,
                                # 'bus_stop_id': 'BusStop_A', # 必要に応じて追加
                            }
                            
                            # コレクション 'bus_congestion_data' にドキュメントを追加
                            # add() を使うと自動生成IDのドキュメントが作成される
                            doc_ref = db.collection(FIRESTORE_COLLECTION_NAME).add(data_to_firestore)
                            print(f"{CYAN}[Faster R-CNN]{RESET}     {GREEN}Firestoreにデータをアップロードしました。ドキュメントID: {doc_ref[1].id}{RESET}")
                        except Exception as fb_e:
                            print(f"{CYAN}[Faster R-CNN]{RED}     Firestoreへのデータアップロード中にエラーが発生しました: {fb_e}{RESET}")
                    else:
                        print(f"{CYAN}[Faster R-CNN]{RED}     Firebaseクライアントが初期化されていないため、Firestoreへのアップロードはスキップされました。{RESET}")
                    # --- Firebase Cloud Firestore へのデータアップロード処理 終了 ---

                    output_filepath = os.path.join(OUTPUT_PROCESSED_FOLDER, f"processed_{filename}")
                    cv2.imwrite(output_filepath, annotated_img)
                    print(f"{CYAN}[Faster R-CNN]{RESET}     処理済み画像を保存しました: {output_filepath}")

                    # os.remove(filepath) # 必要であれば元画像を削除
                    # print(f"{CYAN}[Faster R-CNN]{RESET}     元の画像を削除しました: {filepath}")

                    processed_files.add(filename)

                except Exception as e:
                    print(f"{CYAN}[Faster R-CNN]{RED}   エラー発生: {filename} の処理中にエラー: {e}。このファイルはスキップします。{RESET}")
                    processed_files.add(filename) 
            
            time.sleep(PROCESS_INTERVAL_FASTER_RCNN)

    except Exception as e:
        print(f"{CYAN}[Faster R-CNN]{RED} エラーによりFaster R-CNN処理タスクが停止しました: {e}{RESET}")

# --- メイン実行部分 ---
if __name__ == "__main__":
    print(f"{BOLD}--- バス混雑検知システムを起動します ---{RESET}")
    print(f"{BLUE}QuickTime PlayerでiPadの画面をミラーリングし、指定領域に配置してください。{RESET}")
    print(f"{BLUE}スクリプトを開始するには Enter を押してください。終了するには Ctrl+C を押してください。{RESET}")
    input() # ユーザーがEnterを押すまで待機

    capture_thread = threading.Thread(target=capture_and_save_screenshot_task)
    faster_rcnn_process_thread = threading.Thread(target=process_images_with_fasterrcnn_task)

    capture_thread.daemon = True
    faster_rcnn_process_thread.daemon = True

    capture_thread.start()
    faster_rcnn_process_thread.start()

    try:
        while True:
            time.sleep(1) 
    except KeyboardInterrupt:
        print(f"\n{BOLD}--- システムを停止します。しばらくお待ちください... ---{RESET}")
    except Exception as e:
        print(f"{RED}メインスレッドで予期せぬエラーが発生しました: {e}{RESET}")

    print(f"{BOLD}--- システムが正常に停止しました ---{RESET}")
