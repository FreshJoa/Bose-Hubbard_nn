FROM ubuntu:16.04
RUN apt-get update
RUN apt-get install -y software-properties-common

RUN add-apt-repository ppa:deadsnakes/ppa \
    && apt update \
    && apt install -y python3.6


RUN apt-get update \
  && apt-get install -y python3-pip python3.6-dev \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3.6 python \
  && pip3 install --upgrade pip
RUN apt-get install -y python3.6-venv

# update pip
RUN python3.6 -m pip install pip --upgrade


ENV VIRTUAL_ENV=/opt/venv
RUN python3.6 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install dependencies:
RUN apt-get update
RUN apt-get -y install gcc
RUN apt-get install -y openmpi-bin
RUN apt-get install -y git
RUN apt install -y libopenmpi-dev
RUN apt-get install -y build-essential cmake libboost-dev libexpat1-dev zlib1g-dev libbz2-dev
RUN apt-get install -y libopenblas-dev



RUN git config --global url."https://${decc5c7835e03b43e6a006b7440678e7e3f5180c}@github.com".insteadOf "ssh://git@github.com"
COPY requirements.txt .

RUN python3.6 -m pip  install -r requirements.txt

RUN mkdir /log

# Run the application:
COPY bose_hubbard_testing.py .


CMD ["./bose_hubbard_testing.py"]
ENTRYPOINT ["python3"]

# sudo docker build -t [docker_image_name] .
# sudo docker save [docker_ID] -o [docker_name].tar # in order to check [docker_ID] execute: sudo docker images
# sudo singularity build [singularity_name].sif docker-archive:[docker_name].tar