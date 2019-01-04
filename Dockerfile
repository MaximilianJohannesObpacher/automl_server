FROM ubuntu:18.04
FROM python:3.6
ENV PYTHONUNBUFFERED 1
RUN mkdir /code
WORKDIR /code
ADD requirements.txt /code/

RUN apt-get update
RUN apt-get install -y swig

RUN pip3 install -U numpy==1.14.5
RUN pip3 install -U Cython
RUN pip3 install -r requirements.txt
ADD . /code/

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh