# Dockerfile otimizado para visão computacional
FROM python:3.11-slim

WORKDIR /app

# Instala dependências do sistema necessárias para OpenCV e pyzbar
RUN apt-get update && apt-get install -y \
    # OpenCV dependencies
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    # pyzbar dependencies  
    libzbar0 \
    # Health check
    curl \
    # Cleanup
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copia requirements primeiro para aproveitar cache do Docker
COPY requirements.txt .

# Instala dependências Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Cria usuário não-root
RUN useradd --create-home --shell /bin/bash app && \
    mkdir -p uploads qr_crops outputs logs && \
    chown -R app:app /app

# Copia código da aplicação
COPY --chown=app:app ./src ./src

# Muda para usuário não-root
USER app

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
