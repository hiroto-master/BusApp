#faster解析
import os
import cv2
import torch
import torchvision
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn
    ,fasterrcnn_mobilenet_v3_large_320_fpn
    ,fasterrcnn_mobilenet_v3_large_fpn
    ,fasterrcnn_resnet50_fpn_v2
)
# ★追加: クラス数を変更するために必要
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor 
from torchvision.transforms import functional as F
from datetime import datetime
import time

from PIL import Image
import numpy as np
import threading
import json 

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

model = { "1":fasterrcnn_mobilenet_v3_large_320_fpn, "2":fasterrcnn_mobilenet_v3_large_fpn, "3":fasterrcnn_resnet50_fpn, "4":fasterrcnn_resnet50_fpn_v2}

# --- 共通設定 ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_IMAGES_FOLDER = os.path.join("Faster-R-CNN", 'input_images')
OUTPUT_PROCESSED_FOLDER = os.path.join("Faster-R-CNN", 'output_images')

# --- ★変更点: 学習済みモデルのパス ---
# Colabで学習してダウンロードした .pth ファイルの名前を指定してください
TRAINED_MODEL_PATH = 'fasterrcnn_bus.pth' 

# --- process_images_with_fasterrcnn.py の設定 ---
CONFIDENCE_THRESHOLD = 0.5
TARGET_CLASSES = [1] 
PROCESS_INTERVAL_FASTER_RCNN = 1 
work_count = 0
total_time = 0
input_model = ""
net = None

def process_images_with_fasterrcnn_task():
    global work_count,total_time,input_model,m,net

    # デバイスの選択: GPUが利用可能ならGPU (CUDA)、なければCPU
    if torch.cuda.is_available:
        while True:
            choose_device = input("使用したいモデルを選択してください CPU or GPU:")
            match choose_device:
                case "CPU":
                    device = torch.device("cpu")
                    break
                case "GPU":
                    device = torch.device('cuda')
                    break
                case _:
                    print("無効な入力です。やり直してください")
    else:
        device = torch.device("cpu")

    print(f"{CYAN}[Faster R-CNN]{RESET} {BLUE}使用デバイス: {device}{RESET}")
    
    print(f"{CYAN}[Faster R-CNN]{RESET} {BLUE}学習モデル ({TRAINED_MODEL_PATH}) をロード中...{RESET}")
    if input_model == "0":# 1. ファイルがあるか確認
        if not os.path.exists(TRAINED_MODEL_PATH):
            print(f"{CYAN}[Faster R-CNN]{RED} エラー: モデルファイル '{TRAINED_MODEL_PATH}' が見つかりません。{RESET}")
            return

    try:
        if input_model == "0":
            # 2. 空のモデルを作成 (重みはNone)
            m = fasterrcnn_resnet50_fpn(weights=None)
            
            # 3. クラス数を学習時と同じ設定に変更 (背景 + person = 2クラス)
            num_classes = 2 
            in_features = m.roi_heads.box_predictor.cls_score.in_features
            m.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
            
            # 4. 学習済みの重みをロード
            # map_locationを使うことで、GPUで学習したモデルをCPUでも読み込めるようにする
            m.load_state_dict(torch.load(TRAINED_MODEL_PATH, map_location=device))
            
            m.to(device) # モデルをデバイスに移動
            m.eval()     # 評価モードに設定
        else:
            net.to(device)
            net.eval()

        print(f"{CYAN}[Faster R-CNN]{RESET} {GREEN}モデルのロードが完了しました。{RESET}")

    except Exception as e:
        print(f"{CYAN}[Faster R-CNN]{RED} モデルのロード中にエラーが発生しました: {e}{RESET}")
        return
    # --- ★変更ここまで ---

    processed_files = set() # 既に処理したファイルを記録するセット

    try:
        while True:
            # フォルダがなければ作成 (エラー防止)
            if not os.path.exists(INPUT_IMAGES_FOLDER):
                os.makedirs(INPUT_IMAGES_FOLDER, exist_ok=True)
            if not os.path.exists(OUTPUT_PROCESSED_FOLDER):
                os.makedirs(OUTPUT_PROCESSED_FOLDER, exist_ok=True)

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
                    start_time = time.time()
                    # OpenCV (BGR) から PIL Image (RGB) に変換し、PyTorchテンソルへ
                    img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
                    img_pil = Image.fromarray(img_rgb)
                    
                    # 画像をPyTorchテンソルに変換 (正規化も含まれる)
                    img_tensor = F.to_tensor(img_pil).to(device)

                    # 推論の実行
                    with torch.no_grad(): 
                        prediction = net([img_tensor])

                    person_count = 0
                    annotated_img = img_np.copy() # 描画用に元の画像をコピー
                    
                    # 検出結果の解析
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
                    
                    
                    save_path = os.path.join(OUTPUT_PROCESSED_FOLDER, f"processed_{filename}")
                    cv2.imwrite(save_path, annotated_img)
                    print(f"{CYAN}[Faster R-CNN]{RESET}     画像を保存しました: {save_path}")
                    
                    processed_files.add(filename)
                    end_time = time.time()
                    work_time = end_time - start_time
                    print(f"処理時間{work_time:.2f}秒")
                    work_count += 1
                    total_time += work_time
                    print(f"平均処理時間:{total_time / work_count:.2f}秒")

                except Exception as e:
                    print(f"{CYAN}[Faster R-CNN]{RED}   エラー発生: {filename} の処理中にエラー: {e}。このファイルはスキップします。{RESET}")
                    processed_files.add(filename) 
            
            time.sleep(PROCESS_INTERVAL_FASTER_RCNN)

    except Exception as e:
        print(f"{CYAN}[Faster R-CNN]{RED} エラーによりFaster R-CNN処理タスクが停止しました: {e}{RESET}")

def config_select():
    global CONFIDENCE_THRESHOLD
    while True:
        input_config = input(f"使用する閾値を半角数字で入力。デフォルト{CONFIDENCE_THRESHOLD}はそのままエンター:")
        if input_config == "":
            print(f"デフォルトの閾値 {CONFIDENCE_THRESHOLD} を使用します。")
            break
        else:
            CONFIDENCE_THRESHOLD = float(input_config)
            print(f"閾値を {CONFIDENCE_THRESHOLD} に設定しました。")
            break

def model_select():
    global TRAINED_MODEL_PATH,m,net,input_model
    while True:
        input_model = input(f"使用するモデルを数字で選択してください(最軽量:1,軽量:2,重め:3,最重:4): ")
        match input_model:
            case "0":
                print(f"裏モデルを選択")
                break
            case "1":
                print(f"最軽量のモデルを選択")
                m = model[input_model]
                net = m(weights = "DEFAULT")
                break
            case "2":
                print(f"軽量のモデルを選択")
                m = model[input_model]
                net = m(weights = "DEFAULT")
                break
            case "3":
                print(f"重めのモデルを選択")
                m = model[input_model]
                net = m(weights = "DEFAULT")
                break
            case "4":
                print(f"最重モデルを選択")
                m = model[input_model]
                net = m(weights = "DEFAULT")
                break
            case _:
                print(f"無効な入力です。やり直してください。")
        
# --- メイン実行部分 ---
if __name__ == "__main__":
    config_select()
    model_select()
    print(f"{BLUE}スクリプトを開始するには Enter を押してください。終了するには Ctrl+C を押してください。{RESET}")
    input() # ユーザーがEnterを押すまで待機

    faster_rcnn_process_thread = threading.Thread(target=process_images_with_fasterrcnn_task)
    faster_rcnn_process_thread.daemon = True
    faster_rcnn_process_thread.start()

    try:
        while True:
            time.sleep(1) 
    except KeyboardInterrupt:
        print(f"\n{BOLD}--- システムを停止します。しばらくお待ちください... ---{RESET}")
    except Exception as e:
        print(f"{RED}メインスレッドで予期せぬエラーが発生しました: {e}{RESET}")

    print(f"{BOLD}--- システムが正常に停止しました ---{RESET}")
