FROM python:3.8
WORKDIR /app
COPY requirements.txt /app/
RUN apt-get update && apt-get install -y libgl1-mesa-glx
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app/
RUN chmod +x /app/script.sh