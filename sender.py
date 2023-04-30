import socket
import os

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
buf = 1024
addr = ("127.0.0.1", 9235)

file_name = 'moose.jpg'

with open(file_name, 'rb') as f:
    data = bytearray(f.read(buf))
    while data:
        if s.sendto(data, addr):
            print(f"sending {data}")
            data = bytearray(f.read(buf))

s.sendto('EOF'.encode(), addr)
s.close()
