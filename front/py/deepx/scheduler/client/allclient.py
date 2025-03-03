from .udpconn import _default_udpconn
from typing import Optional
from deepx.nn import DeepxIR
import time
default_client = _default_udpconn


_id_counter=0
def send(ir:DeepxIR) -> Optional[dict]:
    ir._sent_at=time.time()
    global _id_counter
    _id_counter=_id_counter+1
    ir._id=_id_counter
    resp=default_client.send(str(ir))
