FROM nvidia/cuda:9.0-cudnn7-devel

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    python3-dev \
    python3-pip \
    python3-wheel \
    python3-setuptools \
    libgtk2.0-dev && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

RUN pip3 install -U pip
RUN pip3 install tensorflow-gpu keras numpy scipy matplotlib scikit-learn jupyter h5py

ADD . /root
RUN pip3 install Cython
RUN pip3 install -r /root/requirements.txt

