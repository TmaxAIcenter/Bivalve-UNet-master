FROM nvidia/cuda:11.2.0-devel-ubuntu20.04
ENV TZ=Asia/Seoul
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get -y update
RUN apt-get -y install gcc
RUN apt install -y python3-pip
RUN pip3 install --upgrade pip
RUN apt-get install -y libgl1
RUN apt-get update && apt-get install libglib2.0-0 -y
RUN pip3 install torch==1.10.2+cu113 torchvision==0.11.3+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

RUN mkdir /code
WORKDIR /code
ADD ./requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt

RUN mkdir /model
WORKDIR /model


