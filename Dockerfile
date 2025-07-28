FROM python:3.10-slim

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY save_sbert_model.py .

# Copy model folder early to a temp location
RUN python save_sbert_model.py

# Copy rest of the application
COPY . .

CMD ["python", "app.py"]
