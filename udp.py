#!/usr/bin/env python3
import socket
import os

with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
    s.bind(("127.0.0.1", 9235))
    print("UDP server up and listening")

    addr = ("127.0.0.1", 9235)
    buf = 1024

    data, addr = s.recvfrom(buf)
    print("Received File:", data.strip())
    with open('result_' + data.strip().decode(), 'wb') as f:
        data, _ = s.recvfrom(buf)
        while data:
            f.write(data)
            s.settimeout(5)
            data, addr = s.recvfrom(buf)
