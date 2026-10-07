import paramiko
from scp import SCPClient
from PIL import Image
import time
import os

# --- ラズパイとMacの接続情報を設定してください ---
PI_HOST = "<IP_ADDRESS>"  # ラズパイのIPアドレス  TODO(設定): <IP_ADDRESS>
PI_USER = "raspberrypie"            # ラズパイのユーザー名
PI_PASS = "<PI_PASS>"     # ラズパイのパスワード  TODO(設定): <PI_PASS>

# ラズパイ上で撮影した画像を一時的に保存する場所
pi_image_path = "/home/pi/image.jpg" 
# Macのデスクトップに画像を保存する
mac_image_path = os.path.join(os.path.expanduser("~"), "Desktop", "received_image.jpg")
# ----------------------------------------------------

def main():
    """メインの処理を行う関数"""
    try:
        # 1. ラズパイにSSHで接続
        print(f"ラズパイ ({PI_HOST}) に接続しています...")
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(PI_HOST, username=PI_USER, password=PI_PASS)
        print("接続成功！")

        # 2. ラズパイでカメラ撮影コマンドを実行
        # rpicam-still は最新のラズパイOSで使われるコマンドです
        capture_command = f"rpicam-still -o {pi_image_path} --timeout 1000"
        print("カメラで撮影します...")
        stdin, stdout, stderr = ssh.exec_command(capture_command)
        
        # コマンドの実行完了を待つ
        stdout.channel.recv_exit_status()
        print("撮影完了！")

        # 3. 撮影した画像をラズパイからMacに転送 (SCP)
        print("画像をMacに転送しています...")
        with SCPClient(ssh.get_transport()) as scp:
            scp.get(pi_image_path, mac_image_path)
        print(f"転送完了！画像はデスクトップに保存されました: {mac_image_path}")

        # 4. 転送された画像をMacで表示
        print("画像を表示します。")
        img = Image.open(mac_image_path)
        img.show() # Macのデフォルト画像ビューア（プレビュー.appなど）で開く

    except Exception as e:
        print(f"エラーが発生しました: {e}")
    finally:
        if 'ssh' in locals() and ssh.get_transport().is_active():
            ssh.close()
            print("接続を閉じました。")

if __name__ == "__main__":
    main()
