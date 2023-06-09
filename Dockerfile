FROM mysql/mysql-server
ADD schema.sql /docker-entrypoint-initdb.d
EXPOSE 3306
#
#FROM python:latest
#COPY requirements.txt requirements.txt
#RUN pip install --no-cache-dir -r requirements.txt
#COPY main.py main.py
#COPY udp.py udp.py
#COPY script.sh script.sh
#CMD ./script.sh