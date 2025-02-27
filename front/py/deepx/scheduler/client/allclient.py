from .udpconn import _default_udpconn
from typing import Optional

default_client = _default_udpconn

def send(ir: str) -> Optional[dict]:
    return default_client.send(ir)
