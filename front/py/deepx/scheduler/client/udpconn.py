import socket
from typing import Optional

class UDPConn:
    def __init__(self, endpoint: str = "localhost:8080"):
        # 解析endpoint
        self._host, port_str = endpoint.split(':')
        self._port = int(port_str)
        # 创建UDP socket
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    def send(self, ir: str) -> Optional[dict]:
        """
        通过UDP发送IR字符串
        
        Args:
            ir: IR字符串，如 "add@float32 t1 t2 -> t3"
        """
        try:
            # 将IR字符串编码为bytes并发送
            data = ir.encode('utf-8')
            self._sock.sendto(data, (self._host, self._port))
            return {"status": "ok"}
            
        except Exception as e:
            print(f"发送IR失败: {e}")
            return None

    def __del__(self):
        """确保socket正确关闭"""
        if hasattr(self, '_sock'):
            self._sock.close()

# 全局单例实例
_default_udpconn = UDPConn()
