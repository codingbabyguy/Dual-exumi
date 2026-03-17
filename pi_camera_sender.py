import cv2
import socket
import struct
import pickle

# ================= 配置 =================
SERVER_IP = '172.20.10.4'  # ⬅️ 填入你电脑在热点下的 IPv4
PORT = 9999
# =======================================

def start_client():
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        client_socket.connect((SERVER_IP, PORT))
        print("✅ 已连接到电脑")
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        return

    # 使用 CAP_V4L2 适配新系统驱动
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    
    # 稍微等摄像头稳定（热机）
    import time
    time.sleep(1)

    ret, frame = cap.read()
    if ret:
        # 1. 压缩图片（质量 50%），手机热点传输极快
        _, img_encode = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
        data = pickle.dumps(img_encode)
        
        # 2. 发送
        message = struct.pack("Q", len(data)) + data
        client_socket.sendall(message)
        print("✅ 照片已发送！")
    else:
        print("❌ 无法读取摄像头画面，请检查硬件连接或权限")

    cap.release()
    client_socket.close()

if __name__ == '__main__':
    start_client()