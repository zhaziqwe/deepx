import socket

def send_udp_message(ip, port, message):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.sendto(message.encode(), (ip, port))

argset_str = """
name: argset
dtype: int32
args:
    - 2
    - 3
    - 4
returns:
    - shape1
"""
send_udp_message("localhost", 8080, argset_str) 
new_str = """
name: newtensor
dtype: float32
args:
    - tensor
    - shape1
"""
send_udp_message("localhost", 8080, new_str)
