from .udpconn import _default_udpconn
from typing import Optional
from deepx.nn import DeepxIR
import time
default_client = _default_udpconn

def send(ir:DeepxIR) -> Optional[dict]:
    ir._sent_at=time.time()
    return default_client.send(str(ir))
