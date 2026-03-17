import socket
import cv2
import pickle
import struct

# ================= 配置 =================
# 0.0.0.0 表示监听电脑上所有的网卡接口
HOST_IP = '0.0.0.0'
PORT = 9999
# =======================================

def start_server():
    # 1. 创建 Socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((HOST_IP, PORT))
    server_socket.listen(5)
    print(f"👀 服务端已启动，正在监听端口 {PORT}，等待树莓派连接...")

    # 2. 接受连接
    client_socket, addr = server_socket.accept()
    print(f"✅ 连接成功！来自: {addr}")

    data = b""
    # Q: unsigned long long (8 bytes) - 用于存放数据包的大小
    payload_size = struct.calcsize("Q")

    try:
        while True:
            # 3. 接收数据包头（这一帧图像有多大？）
            while len(data) < payload_size:
                packet = client_socket.recv(4*1024) # 4K buffer
                if not packet: break
                data += packet
            
            if not data: break

            packed_msg_size = data[:payload_size]
            data = data[payload_size:]
            msg_size = struct.unpack("Q", packed_msg_size)[0]

            # 4. 接收这一帧完整的图像数据
            while len(data) < msg_size:
                data += client_socket.recv(4*1024) # 继续接收剩余数据

            frame_data = data[:msg_size]
            data = data[msg_size:]

            # 5. 解码并显示
            frame = pickle.loads(frame_data)
            
            # 显示画面
            cv2.imshow("Raspberry Pi Stream", frame)
            
            # 按 'q' 退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"连接断开: {e}")
    finally:
        client_socket.close()
        server_socket.close()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    start_server()