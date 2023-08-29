FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3-pip \
&& rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

COPY server/requirements.txt /app/
WORKDIR /app

RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip3 install --no-cache-dir -r requirements.txt

RUN git clone https://github.com/timdettmers/bitsandbytes.git /tmp/bitsandbytes && \
    cd /tmp/bitsandbytes && \
    CUDA_VERSION=118 make cuda11x && \
    python3 setup.py install && \
    cd / && rm -rf /tmp/bitsandbytes

COPY server /app

CMD ["python3", "main.py"]