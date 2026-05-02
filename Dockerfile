FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV BLENDER_BIN=/usr/bin/blender
ENV BLENDER_SCRIPT_PATH=blender_script.py
ENV BLENDER_OUTPUT_DIR=media/blender
ENV CAD_PRO_DIR=media/cad_pro

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        blender \
        libgl1 \
        libglib2.0-0 \
        libxext6 \
        libxrender1 \
        libsm6 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p media/blender media/generated media/cad_pro media/data

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000}"]
