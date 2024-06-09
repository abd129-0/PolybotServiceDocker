FROM python:3.8.12-slim-buster
WORKDIR /home/ubuntu
WORKDIR /app
RUN pip install --upgrade pip
COPY . .

RUN pip install -r requirements.txt

CMD ["python3", "app.py"]
