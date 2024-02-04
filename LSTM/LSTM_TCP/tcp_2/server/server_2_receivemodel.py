"""
作者：赵江同
日期：2024年02月01日，14时：06分
"""
import socket
from tensorflow.keras.models import load_model

SERVER_IP = '127.0.0.1'
SERVER_PORT = 8888
MAX_BUF_SIZE = 1024

def receive_model(filename):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((SERVER_IP, SERVER_PORT))
    server_socket.listen(1)

    print("Waiting for connection...")
    conn, addr = server_socket.accept()
    print("Connected with", addr)

    with open(filename, 'wb') as file:
        while True:
            data = conn.recv(MAX_BUF_SIZE)
            if not data:
                break
            file.write(data)

    print("Model received successfully")
    conn.close()

def main():
    receive_model('received_model.h5')
    loaded_model = load_model('received_model.h5')
    # 使用加载的模型进行后续操作

if __name__ == "__main__":
    main()
