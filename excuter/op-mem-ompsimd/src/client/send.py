import socket

def send_udp_message(ip, port, message):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.sendto(message.encode(), (ip, port))

yaml_str = """
op: relu
args:
    - tensor
returns:
    - result
"""
send_udp_message("localhost", 8080, yaml_str)
