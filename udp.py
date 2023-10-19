#!/bin/bash
import socket
from PIL import Image
from predictor import process_photo, stop_processing
from log import log

with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
    s.bind(("127.0.0.1", 9235)) #envs
    print("UDP server up and listening")
    log('UDP\nUDP server up and listening')
    buf = 1024 #envs

    f = []

    while True:
        data, _ = s.recvfrom(buf)
        while data:
            try:
                if data.strip().decode() == 'EOF':
                    process_photo(Image.open(b''.join(f)), 1, 'moose')
                    print('File downloaded')
                    log('UDP\nFile downloaded')
                    f = []
            except UnicodeDecodeError as e:
                pass
            f.append(data)
            log('UDP\nReceiving file')
            data, addr = s.recvfrom(buf)
        else:
            stop_processing(1, 'forest')


