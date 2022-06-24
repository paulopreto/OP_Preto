FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04

RUN echo "Installing dependencies..." && \
    apt-get -y --no-install-recommends update && \
    apt-get -y --no-install-recommends upgrade && \
    apt-get install -y --no-install-recommends \
        build-essential curl cmake git libatlas-base-dev libprotobuf-dev libleveldb-dev libsnappy-dev libhdf5-serial-dev protobuf-compiler libboost-all-dev libgflags-dev libgoogle-glog-dev liblmdb-dev pciutils python3-setuptools python3-dev python3-pip opencl-headers ocl-icd-opencl-dev libviennacl-dev libcanberra-gtk-module libopencv-dev ffmpeg && \
    apt-get autoclean && apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* ~/.cache

# Install Miniconda
RUN curl -so /miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-4.5.11-Linux-x86_64.sh && \
    chmod +x /miniconda.sh && \
    /miniconda.sh -b -p /miniconda && \
    rm /miniconda.sh && \
    /miniconda/bin/conda clean -tipsy && \
    ln -s /miniconda/etc/profile.d/conda.sh /etc/profile.d/conda.sh

ENV PATH=/miniconda/bin:$PATH

# Create a Python 3.6 environment
RUN /miniconda/bin/conda install -y conda-build && \
    /miniconda/bin/conda create -y --name py36 python=3.6 && \
    /miniconda/bin/conda clean -ya

ENV CONDA_DEFAULT_ENV=py36
ENV CONDA_PREFIX=/miniconda/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH
ENV CONDA_AUTO_UPDATE_CONDA=false

RUN echo ". /miniconda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate $CONDA_DEFAULT_ENV" >> ~/.bashrc

RUN pip install -U pip

RUN pip install \
    numpy \
    pandas \    
    protobuf \
    opencv-python

RUN echo "Downloading and building OpenPose..." && \
    git clone --branch v1.6.0 https://github.com/CMU-Perceptual-Computing-Lab/openpose.git && \
    mkdir -p /openpose/build && \
    cd /openpose/build && \
    cmake .. && \
    make -j`nproc`

WORKDIR /openpose

COPY USER_ID.txt /openpose/
COPY execute.sh /openpose/
COPY json2dvideow.py /openpose/

# -- define the image's main command
ENTRYPOINT ["/bin/bash", "execute.sh"]
