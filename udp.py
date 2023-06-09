#!/usr/bin/env python3.9
import socket
from datetime import datetime
import mysql.connector
import os

with mysql.connector.connect(
  host="localhost", #envs
  user="root",
  password="password"
) as mydb:

    with mydb.cursor() as cur:
        cur.execute("USE DB")

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
                            print('File Received')
                            # а тут позвать для каждого кадра метод предсказания и выбрать лучший
                            prediction_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            sql_stmt = f"INSERT INTO prediction(prediction_time, photo) VALUES(%s,%s)"
                            cur.execute(sql_stmt, (prediction_time, b''.join(f)))
                            mydb.commit()
                            print('File downloaded')
                            f = []
                    except UnicodeDecodeError as e:
                        pass
                    f.append(data)
                    data, addr = s.recvfrom(buf)


