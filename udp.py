#!/usr/bin/env python3.9
import socket
from PIL import Image
from predictor import process_photo, stop_processing

with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
    s.bind(("127.0.0.1", 9235)) #envs
    print("UDP server up and listening")
    buf = 1024 #envs

    f = []

    while True:
        data, _ = s.recvfrom(buf)
        while data:
            try:
                if data.strip().decode() == 'EOF':
                    process_photo(Image.open(b''.join(f)), 1)
                    print('File downloaded')
                    f = []
            except UnicodeDecodeError as e:
                pass
            f.append(data)
            data, addr = s.recvfrom(buf)
        else:
            stop_processing(1)


