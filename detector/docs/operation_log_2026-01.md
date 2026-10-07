# 実運用ログ（2026年1月）

ラズパイ2台を実際に動かしたときのログの要約と冒頭部分。元のログはここには入れていない。
時刻はラズパイの時計のまま（タイムゾーンは未確認。UTC の場合は +9 時間で日本時間）。

## 親機 Pi B（`log_b.txt`）

- 行数: 8,971
- 期間（画像ファイル名の時刻）: 2026-01-09 〜 2026-01-16
- CPU 温度: 最高 57.9°C / 平均 35.5°C（5,019 回計測）
- Firestore への書き込み: 552 回、合計人数の最大 66 人
- Mac への画像転送失敗: 508 回
- Firestore 書き込み失敗: 4 回

冒頭 40 行（IP・ユーザー名などは伏せ字）:

```text
ネットワーク接続待機中...
ネットワーク接続確認！

[親機B] 起動完了！Wi-Fi接続OK。
Gmail送信成功: 【Pi B】起動成功
[Main] Firebase initialized.
Mac folders checked/created.
[Main] Loading YOLO model from /home/raspberrypie/bus_congestion_project/best_2.pt...
[Main] Model loaded.
[Monitor] 監視スレッドを開始します。
[Main] All threads started. (Ctrl+C to stop)
[Server] Listening for Pi A on port 5000...
[Cam 0] Processing started.
[02:50:08] [Monitor] Temp: 47.4°C | Total: 0 (Local: 0, Remote: 0)
[02:50:10] [Monitor] Temp: 45.8°C | Total: 0 (Local: 0, Remote: 0)
[02:50:12] [Monitor] Temp: 45.2°C | Total: 0 (Local: 0, Remote: 0)
[02:50:14] [Monitor] Temp: 48.0°C | Total: 0 (Local: 0, Remote: 0)
[02:50:16] [Monitor] Temp: 50.1°C | Total: 0 (Local: 0, Remote: 0)
File sent to Mac: /Users/<MAC_USER>/Desktop/raw_images/original_cam0_20260109025006.jpg
[02:50:18] [Monitor] Temp: 48.0°C | Total: 0 (Local: 0, Remote: 0)
[Firestore] Uploaded Total: 0
[02:50:20] [Monitor] Temp: 47.4°C | Total: 0 (Local: 0, Remote: 0)
[02:50:22] [Monitor] Temp: 46.3°C | Total: 0 (Local: 0, Remote: 0)
[02:50:24] [Monitor] Temp: 49.6°C | Total: 0 (Local: 0, Remote: 0)
[02:50:26] [Monitor] Temp: 50.7°C | Total: 0 (Local: 0, Remote: 0)
File sent to Mac: /Users/<MAC_USER>/Desktop/raw_images/original_cam0_20260109025023.jpg
[Firestore] Uploaded Total: 0
[02:50:28] [Monitor] Temp: 48.5°C | Total: 0 (Local: 0, Remote: 0)
[02:50:30] [Monitor] Temp: 47.4°C | Total: 0 (Local: 0, Remote: 0)
[02:50:32] [Monitor] Temp: 46.9°C | Total: 0 (Local: 0, Remote: 0)
[02:50:34] [Monitor] Temp: 50.7°C | Total: 0 (Local: 0, Remote: 0)
[02:50:36] [Monitor] Temp: 49.6°C | Total: 0 (Local: 0, Remote: 0)
File sent to Mac: /Users/<MAC_USER>/Desktop/raw_images/original_cam0_20260109025032.jpg
[Firestore] Uploaded Total: 0
[02:50:38] [Monitor] Temp: 49.6°C | Total: 0 (Local: 0, Remote: 0)
[02:50:40] [Monitor] Temp: 51.2°C | Total: 0 (Local: 0, Remote: 0)
[02:50:42] [Monitor] Temp: 52.9°C | Total: 0 (Local: 0, Remote: 0)
[02:50:44] [Monitor] Temp: 52.4°C | Total: 0 (Local: 0, Remote: 0)
File sent to Mac: /Users/<MAC_USER>/Desktop/raw_images/original_cam0_20260109025041.jpg
[Firestore] Uploaded Total: 0
```

## 子機 Pi A（`log_a.txt`）

- 行数: 8,978
- 期間（画像ファイル名の時刻）: 2026-01-09 〜 2026-01-13
- CPU 温度: 最高 53.5°C / 平均 34.4°C（5,688 回計測）
- 検出: 632 回、最大 39 人
- Mac への画像転送失敗: 476 回
- 親機への人数送信失敗: 121 回
- 例外（Traceback）: 3 回

冒頭 40 行（IP・ユーザー名などは伏せ字）:

```text
/usr/bin/python3: can't open file '/home/raspberrypi/bus_congestion_project/pi_a_final_client.py': [Errno 2] No such file or directory
Traceback (most recent call last):
  File "/home/raspberrypi/bus_congestion_project/congestion_se.py", line 1, in <module>
    import cv2
ModuleNotFoundError: No module named 'cv2'
ネットワーク接続待機中...
接続失敗...2秒後に再試行
接続失敗...2秒後に再試行
接続失敗...2秒後に再試行
接続失敗...2秒後に再試行
接続失敗...2秒後に再試行
接続失敗...2秒後に再試行
接続失敗...2秒後に再試行
接続失敗...2秒後に再試行
接続失敗...2秒後に再試行
接続失敗...2秒後に再試行
接続失敗...2秒後に再試行
接続失敗...2秒後に再試行
接続失敗...2秒後に再試行
接続失敗...2秒後に再試行
接続失敗...2秒後に再試行
ネットワーク接続確認！

[子機A] 起動完了！Wi-Fi接続OK。
Gmail送信成功: 【Pi A】起動成功
[Main] Loading YOLO model from /home/raspberrypi/bus_congestion_project/best_2.pt...
[Monitor] 監視スレッドを開始します。
[Main] Pi A Client started. Press Ctrl+C to stop.
[Cam 0] Processing started.
[Monitor] Temp: 46.3°C | Mem:  13.5%
[Monitor] Temp: 48.5°C | Mem:  14.3%
[Cam 0] Detected: 1 persons.
[Monitor] Temp: 44.6°C | Mem:  14.2%
[Socket] 親機への送信失敗: timed out
File sent to Mac: /Users/<MAC_USER>/Desktop/raw_images/original_camA_20260109022745.jpg
[Monitor] Temp: 45.8°C | Mem:  14.0%
File sent to Mac: /Users/<MAC_USER>/Desktop/received_images/annotated_camA_20260109022745.jpg
[Monitor] Temp: 44.1°C | Mem:  13.7%
[Monitor] Temp: 44.1°C | Mem:  13.3%
[Monitor] Temp: 46.9°C | Mem:  13.4%
```
