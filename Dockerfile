FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY appv2.py .
COPY .streamlit .streamlit

EXPOSE 7860

CMD ["streamlit", "run", "appv2.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.maxUploadSize=200"]
