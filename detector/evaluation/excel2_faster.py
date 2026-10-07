import os
import cv2
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image
from datetime import datetime
import time
import openpyxl

# --- 設定 ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# ユーザー指定のパスに合わせています
INPUT_FOLDER = os.path.join("yolo_9", 'input_images') 
OUTPUT_FOLDER_PROCESSED = os.path.join("yolo_9", 'output_images2')
EXCEL_FILE_PATH = os.path.join("yolo_9", 'people_count_log2.xlsx')

# 検証に使用するモデル名（ログ用）
MODEL_NAME = "fasterrcnn_mobilenet_v3_large_fpn"

# COCOデータセットにおける 'person' のIDは 1 (YOLOは0)
TARGET_CLASS_ID = 1 

# --- フォルダの準備 ---
if not os.path.exists(OUTPUT_FOLDER_PROCESSED):
    os.makedirs(OUTPUT_FOLDER_PROCESSED)
    print(f"出力フォルダを作成しました: {OUTPUT_FOLDER_PROCESSED}")

# --- Excelへの書き込み関数 ---
def save_to_excel(timestamp, filename, count, conf_threshold):
    try:
        if os.path.exists(EXCEL_FILE_PATH):
            wb = openpyxl.load_workbook(EXCEL_FILE_PATH)
            ws = wb.active
        else:
            wb = openpyxl.Workbook()
            ws = wb.active
            ws.title = "Experiment Log"
            ws.append(["Threshold", "Timestamp", "Filename", "Person Count", "Model"])
            print(f"新規Excelファイルを作成しました: {EXCEL_FILE_PATH}")

        # データを追記
        ws.append([conf_threshold, timestamp, filename, count, MODEL_NAME])
        wb.save(EXCEL_FILE_PATH)

    except PermissionError:
        print("    【エラー】Excelファイルが開かれています。閉じてから再実行してください。")
        time.sleep(1)
    except Exception as e:
        print(f"    Excel保存エラー: {e}")

# --- 空行を入れる関数 ---
def add_blank_row():
    try:
        if os.path.exists(EXCEL_FILE_PATH):
            wb = openpyxl.load_workbook(EXCEL_FILE_PATH)
            ws = wb.active
            ws.append([]) 
            wb.save(EXCEL_FILE_PATH)
    except Exception:
        pass

def run_experiment():
    # デバイス設定 (GPUがあれば使う)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用デバイス: {device}")

    print(f"モデルをロード中: {MODEL_NAME}...")
    
    # --- モデルのロード ---
    # 標準の学習済みモデル (ResNet50) を使用
    # ※もし独自学習モデル(.pth)を使いたい場合は、ここで load_state_dict してください
    weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)
    model.to(device)
    model.eval() # 評価モード (必須)
    print("モデルのロードが完了しました。")

    if not os.path.exists(INPUT_FOLDER):
        print(f"入力フォルダが見つかりません: {INPUT_FOLDER}")
        return

    image_files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        print("処理する画像が見つかりません。")
        return

    print(f"検証対象画像: {len(image_files)}枚")

    # --- しきい値を 10 (1.0) から 1 (0.1) までループ ---
    for i in range(10, 0, -1):
        current_conf = i / 10.0
        
        print(f"\n{'='*40}")
        print(f"検証開始: しきい値 (Conf) = {current_conf}")
        print(f"{'='*40}")

        processed_count = 0

        for filename in sorted(image_files):
            filepath = os.path.join(INPUT_FOLDER, filename)
            
            try:
                # 1. 画像読み込みと前処理
                img_cv = cv2.imread(filepath)
                if img_cv is None: continue

                # PyTorch用に変換 (BGR -> RGB -> Tensor)
                img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)
                img_tensor = F.to_tensor(img_pil).to(device)
                
                # バッチ次元を追加 [C, H, W] -> [1, C, H, W]
                img_tensor = img_tensor.unsqueeze(0)

                # 2. 推論実行 (Faster R-CNNは引数でconfを指定できないので、後で手動フィルタリングする)
                with torch.no_grad():
                    prediction = model(img_tensor)

                # 3. 結果の解析とフィルタリング
                # prediction[0] が結果の辞書
                boxes = prediction[0]['boxes'].cpu().numpy()
                labels = prediction[0]['labels'].cpu().numpy()
                scores = prediction[0]['scores'].cpu().numpy()

                person_count = 0
                annotated_img = img_cv.copy()

                # 検出されたすべてのボックスを確認
                for idx in range(len(boxes)):
                    score = scores[idx]
                    label = labels[idx]

                    # ★ここが重要: 手動でしきい値判定を行う
                    # クラスID 1 が person
                    if label == TARGET_CLASS_ID and score >= current_conf:
                        person_count += 1
                        
                        # 描画
                        x1, y1, x2, y2 = map(int, boxes[idx])
                        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(annotated_img, f'{score:.2f}', (x1, y1 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Excelへ記録
                current_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                save_to_excel(current_time_str, filename, person_count, current_conf)

                # 画像保存
                output_filename = f"processed_conf{current_conf}_{filename}"
                output_filepath = os.path.join(OUTPUT_FOLDER_PROCESSED, output_filename)
                cv2.imwrite(output_filepath, annotated_img)
                
                processed_count += 1

            except Exception as e:
                print(f"  エラー発生: {filename} の処理中にエラー: {e}")
            
            # 元のコードに合わせて、画像ごとに空行を入れたい場合はここで呼び出します
            # add_blank_row() 
        
        # しきい値の区切りごとに空行を入れるのが一般的です
        add_blank_row()
        print(f"  >>> しきい値 {current_conf} の処理完了")

    print("\nすべての検証が終了しました。")

if __name__ == "__main__":
    print("--- Faster R-CNN 検証モード ---")
    print(f"対象フォルダ: {INPUT_FOLDER}")
    print(f"出力先: {OUTPUT_FOLDER_PROCESSED}")
    print("しきい値を 1.0 -> 0.1 まで変化させて記録します。")
    print("開始するには Enter を押してください...")
    try:
        input()
        start_time = time.time()
        run_experiment()
        end_time = time.time()
        print(f"所要時間: {end_time - start_time:.1f}秒")
    except KeyboardInterrupt:
        print("\n中断しました。")
        exit()
