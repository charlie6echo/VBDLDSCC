FROM tensorflow/tensorflow:1.15.2-py3

RUN apt-get update \
&& apt install -y libgl1-mesa-glx \
&& apt-get install -y git \
&& apt install -y vim \
&& pip install --upgrade  pip 
