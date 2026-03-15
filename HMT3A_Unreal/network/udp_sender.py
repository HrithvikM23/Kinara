import socket
from config import UDP_IP, UDP_PORT


class UDPSender:

    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        print(f"UDPSender ready → {UDP_IP}:{UDP_PORT}")

    def send(self, data: bytes):
        self.sock.sendto(data, (UDP_IP, UDP_PORT))

    def close(self):
        self.sock.close()
        print("UDPSender closed.")