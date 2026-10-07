#efficiendet.py
import os
import cv2
import torch
import torchvision
# from torchvision.models.detection import detr_resnet50 # DETRは不要
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
import timm # timmライブラリをインポート
from timm.data import resolve_model_data_config
from timm.data.transforms_factory import create_transform

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

# --- process_images_with_efficientdet.py の設定 ---
CONFIDENCE_THRESHOLD = 0.4 # EfficientDetはYOLOより低めの閾値でも良い場合がある
# COCOデータセットにおける 'person' のIDは 1 です。
# timmのEfficientDetはCOCO 90クラスのサブセットに対応している場合があるので、
# モデルの出力クラスとCOCO_INSTANCE_CATEGORY_NAMESを照合して確認が必要です。
# 通常は COCO ID そのままですが、N/Aクラスが除外されるなどしてIDがずれる可能性も考慮。
# ここでは一旦COCO ID = 1 (person) とします。
TARGET_CLASSES = [1] 
PROCESS_INTERVAL_EFFICIENTDET = 5 # 新しい画像を監視する間隔（秒）

# COCOデータセットのクラス名リスト (確認用)
# EfficientDetが内部で持つクラス名リストと正確に一致するかは、
# モデルの実装によって異なる場合があります。
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog',
    'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A',
    'dining table', 'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'N/A', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


# --- Firebase 設定 ---
FIREBASE_SERVICE_ACCOUNT_KEY_PATH = os.path.join(SCRIPT_DIR, 'serviceAccountKey.json')  # TODO(設定): serviceAccountKey.json
FIRESTORE_COLLECTION_NAME = 'bus_congestion_data' 

# --- フォルダの準備 ---
if not os.path.exists(INPUT_IMAGES_FOLDER):
    os.makedirs(INPUT_IMAGES_FOLDER)
    print(f"{GREEN}入力画像フォルダを作成しました: {INPUT_IMAGES_FOLDER}{RESET}")

if not os.path.exists(OUTPUT_PROCESSED_FOLDER):
    os.makedirs(OUTPUT_PROCESSED_FOLDER)
    print(f"{GREEN}処理済み画像フォルダを作成しました: {OUTPUT_PROCESSED_FOLDER}{RESET}")

# --- Firebase の初期化 ---
db = None 
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

# --- process_images_with_efficientdet_task の機能 ---
def process_images_with_efficientdet_task():
    """
    指定フォルダ内の新しい画像をEfficientDetで処理し、結果を描画して保存します。
    検出された人数をJSONデータとしてFirebase Cloud Firestoreにアップロードします。
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"{CYAN}[EfficientDet]{RESET} {BLUE}使用デバイス: {device}{RESET}")
    
    # EfficientDetモデルのロード (timmライブラリを使用)
    # モデル名 'tf_efficientdet_d0' はEfficientDet-D0に相当します。
    # 他のモデルサイズ (d1, d2, d3...) もあります。
    model_name = 'tf_efficientdet_d0' 
    print(f"{CYAN}[EfficientDet]{RESET} {BLUE}EfficientDetモデル ({model_name}) をロード中...{RESET}")
    model = timm.create_model(model_name, pretrained=True, num_classes=len(COCO_INSTANCE_CATEGORY_NAMES))
    # num_classes はCOCOデータセットのクラス数 (80+1背景 = 91) を指定することが多いですが、
    # timmの特定のEfficientDet実装では80クラスで学習されていることもあります。
    # ここではCOCO_INSTANCE_CATEGORY_NAMESの長さを指定していますが、
    # モデルの出力と合わない場合は調整が必要です。
    
    model.to(device)
    model.eval()
    print(f"{CYAN}[EfficientDet]{RESET} {GREEN}モデルのロードが完了しました。{RESET}")

    # モデルに合わせた前処理の準備
    # timmのモデルは、そのモデル用に最適化された前処理設定を持っています
    config = resolve_model_data_config(model, None)
    transform = create_transform(**config, is_training=False)

    processed_files = set()

    try:
        while True:
            all_files = [f for f in os.listdir(INPUT_IMAGES_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            new_files = [f_name for f_name in all_files if f_name not in processed_files]

            if not new_files:
                time.sleep(PROCESS_INTERVAL_EFFICIENTDET)
                continue

            print(f"{CYAN}[EfficientDet]{RESET} {GREEN}新しい画像 {len(new_files)} 件を処理します。{RESET}")

            for filename in sorted(new_files):
                filepath = os.path.join(INPUT_IMAGES_FOLDER, filename)
                print(f"{CYAN}[EfficientDet]{RESET}   処理中: {filename}")

                try:
                    img_np = cv2.imread(filepath)
                    if img_np is None:
                        print(f"{CYAN}[EfficientDet]{RESET} {YELLOW}警告: 画像を読み込めませんでした: {filename}。スキップします。{RESET}")
                        processed_files.add(filename) 
                        continue
                    
                    # OpenCV (BGR) から PIL Image (RGB) に変換
                    img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
                    img_pil = Image.fromarray(img_rgb)
                    
                    # 前処理を適用し、PyTorchテンソルへ
                    img_tensor = transform(img_pil).unsqueeze(0).to(device) # バッチ次元を追加

                    # 推論の実行
                    with torch.no_grad():
                        # EfficientDetの出力はモデルの実装によって異なる場合があります。
                        # 通常は [バッチサイズ, クエリ数, 4(box) + クラス数(score)] のテンソルか、
                        # リスト形式でボックス、スコア、クラスIDが返されます。
                        # timmのEfficientDetは、以下のような形式で返されることが多いです。
                        # (detections) -> shape (batch_size, num_detections, 6)
                        # 6要素: [x1, y1, x2, y2, score, class_id]
                        detections = model(img_tensor) 

                    person_count = 0
                    annotated_img = img_np.copy()

                    # 検出結果の解析
                    # detections は通常 [batch_size, num_detections, 6] のテンソル
                    # 6要素は [x1, y1, x2, y2, score, class_id]
                    # バッチサイズ1なので detections[0] を使う
                    
                    # NMSがモデル内部で行われている場合とそうでない場合があるため、
                    # ここではシンプルに閾値でフィルタリング
                    for det in detections[0]: # 各検出結果についてループ
                        score = det[4].item()
                        label = int(det[5].item())
                        
                        if score >= CONFIDENCE_THRESHOLD and label == TARGET_CLASSES[0]:
                            person_count += 1
                            x1, y1, x2, y2 = map(int, det[:4].cpu().numpy()) # box座標

                            # 座標が画像範囲内に収まるようにクリッピング
                            h, w, _ = img_np.shape
                            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)

                            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2) 
                            cv2.putText(annotated_img, f'Person {score:.2f}', (x1, y1 - 10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    print(f"{CYAN}[EfficientDet]{RESET}     {BOLD}検出された人物の数: {person_count}{RESET}")
                    
                    # --- Firebase Cloud Firestore へのデータアップロード処理 ---
                    if db: 
                        try:
                            data_to_firestore = {
                                'timestamp': datetime.now(), 
                                'detected_people_count': person_count,
                                'source_image_filename': filename,
                            }
                            
                            doc_ref = db.collection(FIRESTORE_COLLECTION_NAME).add(data_to_firestore)
                            print(f"{CYAN}[EfficientDet]{RESET}     {GREEN}Firestoreにデータをアップロードしました。ドキュメントID: {doc_ref[1].id}{RESET}")
                        except Exception as fb_e:
                            print(f"{CYAN}[EfficientDet]{RED}     Firestoreへのデータアップロード中にエラーが発生しました: {fb_e}{RESET}")
                    else:
                        print(f"{CYAN}[EfficientDet]{RED}     Firebaseクライアントが初期化されていないため、Firestoreへのアップロードはスキップされました。{RESET}")
                    # --- Firebase Cloud Firestore へのデータアップロード処理 終了 ---

                    output_filepath = os.path.join(OUTPUT_PROCESSED_FOLDER, f"processed_{filename}")
                    cv2.imwrite(output_filepath, annotated_img)
                    print(f"{CYAN}[EfficientDet]{RESET}     処理済み画像を保存しました: {output_filepath}")

                    # os.remove(filepath) 
                    # print(f"{CYAN}[EfficientDet]{RESET}     元の画像を削除しました: {filepath}")

                    processed_files.add(filename)

                except Exception as e:
                    print(f"{CYAN}[EfficientDet]{RED}   エラー発生: {filename} の処理中にエラー: {e}。このファイルはスキップします。{RESET}")
                    processed_files.add(filename) 
            
            time.sleep(PROCESS_INTERVAL_EFFICIENTDET)

    except Exception as e:
        print(f"{CYAN}[EfficientDet]{RED} エラーによりEfficientDet処理タスクが停止しました: {e}{RESET}")

# --- メイン実行部分 ---
if __name__ == "__main__":
    print(f"{BOLD}--- バス混雑検知システムを起動します ---{RESET}")
    print(f"{BLUE}QuickTime PlayerでiPadの画面をミラーリングし、指定領域に配置してください。{RESET}")
    print(f"{BLUE}スクリプトを開始するには Enter を押してください。終了するには Ctrl+C を押してください。{RESET}")
    input() 

    capture_thread = threading.Thread(target=capture_and_save_screenshot_task)
    efficientdet_process_thread = threading.Thread(target=process_images_with_efficientdet_task)

    capture_thread.daemon = True
    efficientdet_process_thread.daemon = True

    capture_thread.start()
    efficientdet_process_thread.start()

    try:
        while True:
            time.sleep(1) 
    except KeyboardInterrupt:
        print(f"\n{BOLD}--- システムを停止します。しばらくお待ちください... ---{RESET}")
    except Exception as e:
        print(f"{RED}メインスレッドで予期せぬエラーが発生しました: {e}{RESET}")

    print(f"{BOLD}--- システムが正常に停止しました ---{RESET}")
