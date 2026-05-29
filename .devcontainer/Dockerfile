FROM mcr.microsoft.com/devcontainers/python:1-3.11-bullseye

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libtesseract-dev \
    libleptonica-dev \
    git \
    build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /workspaces/local-pdf-qa
