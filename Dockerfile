FROM pytorch/pytorch:2.5.0-cuda12.4-cudnn9-devel

RUN apt-get update && apt-get install -y \
    git wget curl vim build-essential zip \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel

WORKDIR /tmp/berkeley-function-call-leaderboard
COPY schemabench/berkeley-function-call-leaderboard /tmp/berkeley-function-call-leaderboard
RUN pip install -e .

COPY requirements.txt /tmp/requirements.txt
WORKDIR /tmp
RUN pip install -r requirements.txt