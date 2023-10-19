#!/bin/bash

python /app/main.py &
python /app/udp.py &
wait -n
exit $?