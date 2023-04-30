#!/bin/bash

./main.py &
./udp.py &
wait -n
exit $?