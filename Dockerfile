FROM python:3.7

WORKDIR /app

COPY app.py .
COPY tweet_topic_model.pkl .
COPY templates templates/

RUN pip install flask torch transformers

EXPOSE 5000

CMD ["python", "app.py"]