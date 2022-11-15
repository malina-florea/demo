FROM pytorch/pytorch:latest

COPY ./src /app/src
COPY load_models.py /app
WORKDIR /app

RUN pip3 install fastapi uvicorn transformers && \
    python load_models.py

CMD [ "uvicorn", "src.main:app", "--host", "0.0.0.0", "--reload"]
