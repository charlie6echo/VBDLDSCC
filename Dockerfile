FROM tensorflow/tensorflow:1.15.2-gpu

RUN apt-get update \
&& apt install -y libgl1-mesa-glx \
&& apt-get install -y git \
&& apt install -y vim \
&& pip install --upgrade  pip 
