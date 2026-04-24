
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

#  базовые пакетов + Python 3.12  
RUN apt-get update && apt-get install -y \
    software-properties-common \
    curl \
    wget \
    ca-certificates \
    zstd \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    && rm -rf /var/lib/apt/lists/*

# 2.  pip для Python 3.12
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

# aliias
RUN ln -sf /usr/bin/python3.12 /usr/bin/python && \
    ln -sf /usr/bin/python3.12 /usr/bin/python3 && \
    ln -sf /home/a/.local/bin/pip3 /usr/bin/pip || ln -sf /usr/local/bin/pip3 /usr/bin/pip

#  Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

WORKDIR /app

COPY requirements.txt .

# 5. Установка зависимостей через python3.12 -m pip  
RUN python3.12 -m pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

EXPOSE 11434 8000


CMD ["sh", "-c", "ollama serve & sleep 5 && ollama pull qwen2.5:0.5b && uvicorn src.main:app --host 0.0.0.0 --port 8000"]