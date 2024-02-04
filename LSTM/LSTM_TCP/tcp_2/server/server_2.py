"""
作者：赵江同
日期：2024年02月01日，13时：06分
"""
import socket

SERVER_IP = '127.0.0.1'
SERVER_PORT = 8888
MAX_BUF_SIZE = 1024

def main():
    # 创建服务器套接字
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((SERVER_IP, SERVER_PORT))
    server_socket.listen(5)

    print("Server is listening...")

    while True:
        # 等待客户端连接
        client_socket, client_addr = server_socket.accept()

        # 接收客户端发送的数据
        data = client_socket.recv(MAX_BUF_SIZE).decode()
        print("Received data:\n", data)

        # 将接收到的数据保存到文件中
        with open('data_received.csv', 'a') as file:
            file.write(data + '\n')

        print("Data saved to data_received.csv")

        # 关闭客户端套接字
        client_socket.close()

if __name__ == "__main__":
    main()
