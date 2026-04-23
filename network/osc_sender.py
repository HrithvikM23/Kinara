from __future__ import annotations


class OSCSender:
    def __init__(self, host: str = "127.0.0.1", port: int = 9000, enabled: bool = False):
        self.host = host
        self.port = port
        self.enabled = enabled

    def send_pose(self, body_points, hands_by_side) -> None:
        return

    def close(self) -> None:
        return
