from __future__ import annotations

import socket


class UDPSender:
    def __init__(self, udp_ip: str, udp_port: int):
        self.address = (udp_ip, udp_port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        print(f"UDPSender ready -> {udp_ip}:{udp_port}")

    def send(self, data: bytes) -> None:
        self.sock.sendto(data, self.address)

    def close(self) -> None:
        self.sock.close()
        print("UDPSender closed.")
