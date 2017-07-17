FROM debian:stretch-slim

MAINTAINER Misha Wagner <"mishajw@gmail.com">

ADD . /object_net
WORKDIR /object_net

RUN \
  apt-get update -y && \
  apt-get install -y python3-pip && \
  pip3 install -r requirements.txt && \
  apt-get clean -y && \
  rm -rf /var/lib/apt /tmp/* /var/tmp/*

CMD ["python3", "main.py"]

