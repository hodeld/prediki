FROM python:3.6-slim
COPY ./app.py /deploy/
COPY ./requirements.txt /deploy/
COPY ./sentiment.pth /deploy/sentiment.pth/
WORKDIR /deploy/
RUN pip install -r requirements.txt
EXPOSE 80
ENTRYPOINT ["python", "app.py"]
