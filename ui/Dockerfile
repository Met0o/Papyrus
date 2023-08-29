FROM python:3.10-slim-buster

RUN apt-get update && apt-get install -y \
    python3-pip \
&& rm -rf /var/lib/apt/lists/*

COPY ui/requirements.txt /app/
WORKDIR /app

RUN pip3 install --no-cache-dir -r requirements.txt

COPY ui /app

CMD ["streamlit", "run", "webapp.py"]