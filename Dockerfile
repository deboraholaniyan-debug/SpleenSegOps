
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.15.0

RUN pip install --no-cache-dir --index-url https://pypi.org/simple -r requirements.txt

COPY src/ ./src/
COPY external/SlimFormer3D/ ./external/SlimFormer3D/
COPY models/slimformer3d_best.pth ./models/slimformer3d_best.pth

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
