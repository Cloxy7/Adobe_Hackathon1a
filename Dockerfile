FROM --platform=linux/amd64 python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "save_sbert_model.py"]
CMD ["python", "app.py"]