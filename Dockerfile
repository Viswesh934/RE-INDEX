FROM mcr.microsoft.com/devcontainers/python:3.11

# Remove broken Yarn repo that causes Codespaces apt failures
RUN rm -f /etc/apt/sources.list.d/yarn.list

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libtesseract-dev \
    libleptonica-dev \
    poppler-utils \
    git \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

EXPOSE 8501

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt