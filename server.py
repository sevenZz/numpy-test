import socket

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

server_address = ('127.0.0.1', 10086)

sock.bind(server_address)

sock.listen(1)

while True:
    connection, client_address = sock.accept()
    print(connection.recvmsg(1000))
    print(client_address)
