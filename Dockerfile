FROM rayproject/ray-ml:1.7.0-py37-gpu

ARG DEV

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

# RUN apt install libeigen3-dev

RUN git clone https://gitlab.com/libeigen/eigen.git && cd eigen && git checkout 3.4.0
RUN cmake eigen && make install
RUN ln -s /usr/local/include/eigen3/Eigen /usr/local/include/Eigen

RUN wget https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-gpu-linux-x86_64-2.6.0.tar.gz
RUN tar -C /usr/local -xzf libtensorflow-gpu-linux-x86_64-2.6.0.tar.gz
RUN rm libtensorflow-gpu*

WORKDIR /repos

RUN git clone https://github.com/thomas-koller/jass-kit-py.git\
    && cd jass-kit-py && pip install -e . && cd ..

RUN git clone --recurse-submodules \
    https://github.com/thomas-koller/jass-kit-cpp.git\
    && cd jass-kit-cpp && pip install . && cd ..

# required in order for linker to find jass headers
RUN cd jass-kit-cpp && cmake . && make install && cd ..

COPY .github .github
RUN git clone --recurse-submodules \
    https://$(cat .github)@github.com/thomas-koller/jass-ml-cpp.git \
    && cd jass-ml-cpp && pip install . && cd ..

RUN git clone https://$(cat .github)@github.com/thomas-koller/jass-ml-py.git\
    && cd jass-ml-py && pip install -e . && cd ..

COPY requirements.txt requirements.txt
COPY requirements-dev.txt requirements-dev.txt

RUN if [[ -z "$DEV" ]];\
    then echo "No DEV mode";\
    else pip install -r requirements-dev.txt; fi

RUN pip install -r requirements.txt

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

WORKDIR /app

COPY .wandbkey .wandbkey

# add user

RUN chown -hR 1000 /repos

#RUN adduser user --uid 1000
#RUN adduser user sudo
#USER user

ENV XLA_PYTHON_CLIENT_MEM_FRACTION=.7

ENTRYPOINT ["sh", "/entrypoint.sh"]