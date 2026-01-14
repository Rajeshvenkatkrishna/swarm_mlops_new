FROM python:3.10-slim
WORKDIR /app
COPY requirement.txt .
RUN pip install -r requirements.txt
COPY train.py app.py ./
RUN train.py
EXPOSE 5000
CMD ["python", "app.py"]
