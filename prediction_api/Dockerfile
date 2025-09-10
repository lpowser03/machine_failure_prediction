FROM python:3.12-slim
LABEL authors="logan"
RUN pip install --upgrade pip
WORKDIR /app
COPY ../requirements.txt .
RUN pip install -r requirements.txt
COPY app.py model.pkl ./
COPY data/ ./data/
CMD ["python","app.py"]