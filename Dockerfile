FROM rayproject/ray-ml:1.7.0-py37-gpu

WORKDIR /tmp

USER root

# Fix Nvidia repository GPG error, see https://github.com/NVIDIA/nvidia-docker/issues/1632
# see also https://github.com/open-mmlab/mmfashion/issues/147
RUN apt-get install wget -y
RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

RUN apt-get -y update \
    && apt-get -y install \
    curl \
    default-jre \
    git \
    jq \
    python3-opencv \
    python-opencv \
    libfontconfig1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libopenmpi-dev \
    zlib1g-dev \
    graphviz \
    x11-apps \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \

RUN apt-get update && apt-get install build-essential wget git -y

# https://github.com/NVIDIA/nvidia-docker/issues/1632#issuecomment-1112667716
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list

RUN curl -sSL https://cmake.org/files/v3.5/cmake-3.5.2-Linux-x86_64.tar.gz | tar -xzC /opt\
    && apt-get update\
    && wget http://www.cmake.org/files/v3.5/cmake-3.5.2.tar.gz\
    && tar xf cmake-3.5.2.tar.gz\
    && cd cmake-3.5.2\
    &&./configure \
    && make

RUN cd cmake-3.5.2 && make install

RUN git clone https://gitlab.com/libeigen/eigen.git && cd eigen && git checkout 3.4.0
RUN cmake eigen && make install
RUN ln -s /usr/local/include/eigen3/Eigen /usr/local/include/Eigen

RUN wget https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-gpu-linux-x86_64-2.6.0.tar.gz
RUN tar -C /usr/local -xzf libtensorflow-gpu-linux-x86_64-2.6.0.tar.gz
RUN rm libtensorflow-gpu*

RUN chown -hR ray /usr
RUN adduser ray sudo

COPY . /app/
WORKDIR /app

# add user
RUN chown -hR ray /app
USER ray

RUN git config --global --add safe.directory /app

RUN pip install --upgrade pip
RUN pip install --upgrade setuptools

RUN pip install -v --no-cache -e .

RUN pip install -r requirements-dev.txt

RUN chmod +x /app/entrypoint.sh

ENV XLA_PYTHON_CLIENT_MEM_FRACTION=.7

ENTRYPOINT ["sh", "/app/entrypoint.sh"]