# detector — バス停の混雑をカメラと画像認識AIで数える側

BusApp（このリポジトリの Web アプリ）に「今バス停に何人並んでいるか」を送る側のプログラム一式です。
高校のサイエンスAP（探究）で作り、2026年1月にラズパイ2台で実際に動かしました。後輩への引き継ぎ用にまとめています。

- アプリ本体（React）はリポジトリ直下の `src/` です。このフォルダはそれとは独立していて、アプリのビルドには関係しません
- パスワード・メールアドレス・IP アドレスなどは `<MAC_PASSWORD>` のような伏せ字にしてあります。動かすときは [設定が必要な箇所](#設定が必要な箇所) を書き換えてください
- 学習済みモデル・学習データ・撮影画像はこのリポジトリに入れていません（[モデルとデータ](#モデルとデータ)）

## システム全体の流れ

```
 [子機 Pi A]                       [親機 Pi B]                        [Firebase]            [BusApp]
 USBカメラ → YOLO で人数を数える    USBカメラ → YOLO で人数を数える
      │  5秒ごと                        │
      └── 人数を TCP で送信 ──────→ 受信（ポート 5000）
                                        │ 自分の人数 + 子機の人数 = 合計
                                        └── 5秒ごとに書き込み ──→ Firestore ──→ 最新の人数を表示
                                                                  people_count      「何本目のバスに乗れるか」
 （両方とも）起動・ネット切断/復帰を Gmail で通知 / 撮影画像を Mac へ SFTP 転送（画像集め用・任意）
```

- 人の判定: YOLO の検出結果のうち、クラスが `person` で信頼度 0.5 より大きいものを数えています
- アプリ側は「バス1台に 50 人」として、何本目のバスに乗れるかを計算しています（`src/App.js`）

## フォルダ構成

| フォルダ | 中身 |
|---|---|
| `raspberry_pi/parent/` | **本番の親機** `python_a.py` |
| `raspberry_pi/child/` | **本番の子機** `congestion_se.py` |
| `raspberry_pi/single/` | ラズパイ1台だけで動かす版（`one_pi1.py` / `one_pi2.py`。2台構成にする前のもの） |
| `evaluation/` | モデルの比較・検証用。画像フォルダを読んで検出し、結果を Excel や画像で出す |
| `history/` | 本番に至るまでの試作。段階ごとに分けてある（下の表） |
| `docs/` | 手順書（YOLO・Faster R-CNN の学習、アノテーション、ラズパイの IP 固定）と実運用ログの要約 |

### 本番がどれかの根拠

`raspberry_pi/` の2本は、2026年1月9日〜16日の実運用ログに出てくるメッセージ（「[子機A] 起動完了！Wi-Fi接続OK。」「【Pi B】接続成功」など）や、ログ内の実行ファイル名と一致したものです。ログの要約は [docs/operation_log_2026-01.md](docs/operation_log_2026-01.md) にあります。

### history/ の段階

| 段階 | フォルダ | 内容 |
|---|---|---|
| 1 | `history/1_pc_capture/` | PC で画面をキャプチャして検出。Faster R-CNN / YOLO / EfficientDet / DETR / Cascade R-CNN を試した |
| 2 | `history/2_pi_faster_rcnn/` | ラズパイ単体でカメラ撮影 → Faster R-CNN。画像は Mac へ転送 |
| 3 | `history/3_pi_yolo/` | ラズパイ側を YOLO に切り替え（軽くて速い）。CPU 温度・メモリの監視を追加 |
| 4 | `history/4_two_pi/` | 親機・子機の2台構成。`congestion_1.py`(親) / `congestion_2.py`(子) → `force_*.py` → `congestion_ri.py` → 本番 |

ファイルの更新日は整理のときにコピーした日付になっていて、新旧の判断には使えません。順番は中身（機能の増え方）から推定したものです。

### evaluation/ のファイル

| ファイル | 内容 |
|---|---|
| `excel.py` / `excel2.py` | YOLO で画像フォルダを検出し、人数などを Excel に出力 |
| `excel_faster.py` / `excel2_faster.py` | 同じことを Faster R-CNN で |
| `yolo_detect.py` / `yolo_best.py` / `yolo_12.py` / `yolo_9.py` | YOLO で画像フォルダを検出し、枠付き画像を保存 |
| `faster_rcnn_custom.py` | 自分で学習した Faster R-CNN（`fasterrcnn_bus.pth`）で検出。モデルファイルは入れていない |

## 動かし方（本番の2台構成）

1. ラズパイ2台に Raspberry Pi OS を入れ、同じ Wi-Fi（またはスマホのテザリング）につなぐ
2. 親機の IP アドレスを固定する → [docs/raspi_static_ip.md](docs/raspi_static_ip.md)
3. 両方のラズパイで、ホームに `bus_congestion_project` フォルダを作り、スクリプトとモデル `best_2.pt` を置く
   - 親機: `~/bus_congestion_project/python_a.py`
   - 子機: `/home/raspberrypi/bus_congestion_project/congestion_se.py`（パスが直書きなので、ユーザー名が違えば `BASE_DIR` を変える）
4. ライブラリを入れる: `pip install -r requirements-pi.txt`
   - 実運用ログの最初に `No module named 'cv2'` が出ていたので、OpenCV が入っているかを先に確認する
5. Firebase のプロジェクトを作り、サービスアカウントの鍵（JSON）を親機の `bus_congestion_project/serviceAccountKey.json` に置く（下の注意を参照）
6. [設定が必要な箇所](#設定が必要な箇所) を書き換える
7. 親機 → 子機の順に起動する: `python3 python_a.py` / `python3 congestion_se.py`

### Firebase について

- 当時の Firebase プロジェクトは高校の Google アカウントで作っていて、そのアカウントはもう使えません。**新しく作り直す必要があります**
- アプリ側の設定は `src/firebase.js`（空欄になっている）に、Firebase コンソールの「ウェブアプリの設定」を貼り付けます
- 検出側の鍵（サービスアカウントの JSON）は秘密情報です。**絶対に GitHub に上げないでください**（`detector/.gitignore` で除外しています）

### Firestore のデータ構造

| コレクション | 書く側 | 読む側 | フィールド |
|---|---|---|---|
| `people_count` | 親機 `python_a.py`（5秒ごとに add） | アプリ（最新1件をリアルタイム表示） | `count_time`（時刻）, `count`（合計人数）, `detail`（例: `Local:3, Remote:2`） |
| `bus_schedules` | 手で登録（登録用のプログラムは無い） | アプリ（時刻表・直近4本） | `stop_time`（Timestamp。時刻だけ使う）, `bus_label`（路線名）, `id` |

## 設定が必要な箇所

伏せ字の行には `# TODO(設定)` が付いています。`TODO(設定)` で検索すると全部見つかります。本番の2本で書き換えるのは次のとおりです。

| 変数 | 意味 | 親機 `python_a.py` | 子機 `congestion_se.py` |
|---|---|---|---|
| `PI_B_IP_ADDRESS` | 親機の固定 IP アドレス | — | 20行目 |
| `CRED_PATH` | Firebase のサービスアカウント鍵のファイル名 | 24行目 | — |
| `GMAIL_USER` | 通知メールの送信元（Gmail） | 27行目 | 24行目 |
| `GMAIL_APP_PASSWORD` | Gmail の「アプリ パスワード」（通常のパスワードではない） | 28行目 | 25行目 |
| `GMAIL_TO` | 通知メールの送信先 | 29行目 | 26行目 |
| `MAC_IP` / `MAC_USER` / `MAC_PASS` | 画像を受け取る Mac の IP・ユーザー名・パスワード | 32〜34行目 | 29〜31行目 |
| `MAC_DIR_BEFORE` / `MAC_DIR_AFTER` | Mac 側の保存先（`/Users/<MAC_USER>/...`） | 35〜36行目 | 32〜33行目 |

- Gmail 通知と Mac への画像転送は、本番に必須ではありません。使わないときは該当部分を外して構いません（実運用でも Mac への転送は失敗が多かった）
- `history/` と `evaluation/` のファイルも同じ変数名で伏せてあります

## モデルとデータ

| もの | 状態 |
|---|---|
| `best_2.pt`（自分で学習した YOLO、約 39MB） | このリポジトリの Releases に置く予定 |
| `fasterrcnn_bus.pth`（自分で学習した Faster R-CNN） | 本番で使っていないので入れていない |
| 公開済みのモデル（`yolo11m.pt`、`yolov9c.pt` など） | `ultralytics` が初回実行時に自動でダウンロードする |
| 学習データ（Roboflow で作ったデータセット）・撮影画像 | 人が写っているので入れていない |

学習のやり方は [docs/yolo_finetuning_colab.md](docs/yolo_finetuning_colab.md) と [docs/annotation_guide.md](docs/annotation_guide.md) にあります。

## 分かっている課題・続きのアイデア

- Mac への画像転送は失敗が多い（親機で約500回）。本番では外すか、別の保存方法にする
- 子機 → 親機の送信が失敗したとき（約120回）、親機は前回の子機の人数のまま合計を出している
- ラズパイの時計のタイムゾーンを確認していない（ログの時刻が UTC の可能性あり）
- 「バス1台 50 人」はアプリに直書きの仮の値。実際の定員や乗り方に合わせて見直す余地がある
- 実運用ログに、ここに無いファイル（`pi_a_final_client.py`）を実行しようとした跡がある。当時のラズパイの SD カードには、ここより新しい版が残っているかもしれない

## 手順書（docs/）

| ファイル | 内容 |
|---|---|
| [yolo_finetuning_colab.md](docs/yolo_finetuning_colab.md) | Roboflow のデータで YOLO を Google Colab で学習する |
| [faster_rcnn_finetuning_colab.md](docs/faster_rcnn_finetuning_colab.md) | 同じく Faster R-CNN を学習する |
| [annotation_guide.md](docs/annotation_guide.md) | Roboflow でのアノテーション（人に枠を付ける作業） |
| [raspi_static_ip.md](docs/raspi_static_ip.md) | 親機の IP アドレスを固定する（テザリング対応の短い版） |
| [raspi_static_ip_background.md](docs/raspi_static_ip_background.md) | 同じ内容を、なぜ固定が必要かから説明した版 |
| [operation_log_2026-01.md](docs/operation_log_2026-01.md) | 2026年1月の実運用ログの要約 |
