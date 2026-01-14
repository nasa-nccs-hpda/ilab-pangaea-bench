# Arguments to pass to the image
ARG VERSION_DATE=24.01
ARG FROM_IMAGE=nvcr.io/nvidia/pytorch

# Import RAPIDS container as the BASE Image (cuda base image)
FROM ${FROM_IMAGE}:${VERSION_DATE}-py3

# Ubuntu needs noninteractive to be forced
ENV DEBIAN_FRONTEND noninteractive
ENV CPLUS_INCLUDE_PATH="/usr/include/gdal"
ENV C_INCLUDE_PATH="/usr/include/gdal"
ENV GDAL_CONFIG=/usr/bin/gdal-config

# System dependencies
# System dependencies - split into manageable chunks
RUN apt-get update && \
    apt-get -y install software-properties-common && \
    add-apt-repository ppa:ubuntugis/ubuntugis-unstable && \
    apt-get update

RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash

RUN apt-get update && \
    apt-get -y install --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        make \
        wget \
        curl \
        git \
        vim \
        procps \
        bzip2 && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
    apt-get -y install --no-install-recommends \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libfontconfig1 \
        libssl-dev \
        libzmq3-dev \
        libpng-dev \
        libfreetype6-dev \
        libx11-dev \
        locales \
        git-lfs && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
    apt-get -y install --no-install-recommends \
        swig \
        libhdf5-dev \
        libhdf5-serial-dev \
        libhdf4-dev \
        libnetcdf-dev \
        libacl1-dev && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
    apt-get -y install --no-install-recommends \
        gdal-bin \
        libgdal-dev \
        libgeos++-dev \
        libgeos-dev \
        libsqlite3-dev \
        libproj-dev \
        proj-data \
        proj-bin \
        libspatialindex-dev && \
    apt-get -y autoremove && \
    rm -rf /var/cache/apt /var/lib/apt/lists/*

# Install shiftc
WORKDIR /app
RUN git clone --single-branch --branch master https://github.com/pkolano/shift.git && \
    cd shift/c && \
    make nolustre && \
    cd ../ && \
    install -m 755 perl/shiftc /usr/local/bin/ && \
    install -m 755 c/shift-bin /usr/local/bin/ && \
    install -m 755 perl/shift-mgr /usr/local/bin/ && \
    install -m 644 etc/shiftrc /etc/ && \
    install -m 755 perl/shift-aux /usr/local/bin/ && \
    install -m 755 c/shift-bin /usr/local/bin/ && \
    export LC_ALL=en_US.UTF-8 && \
    export LANG=en_US.UTF-8 && \
    locale-gen en_US.UTF-8 && \
    rm -rf /app

# Install core dependencies first
RUN pip install --no-cache-dir 'numpy<2' pytorch-lightning Lightning

# Install data/ML libraries
RUN pip install --no-cache-dir transformers datasets deepspeed webdataset 'huggingface_hub[cli,torch]'

# Install geospatial libraries (these often fail due to C dependencies)
RUN pip install --no-cache-dir torchgeo rasterio rioxarray xarray xarray-spatial geopandas

# Install CV libraries
RUN pip install --no-cache-dir opencv-python-headless tifffile webcolors

# Install remaining packages
RUN pip install --no-cache-dir tiler segmentation-models timm supervision pytest coveralls rtree \
    sphinx sphinx_rtd_theme yacs termcolor segmentation-models-pytorch coverage \
    satpy pyresample netCDF4 h5py goes2go gdown wandb \
    "dask[distributed]" stackstac bokeh pystac-client planetary-computer joblib psutil geotessera

# Install terratorch and git packages last
RUN pip install --no-cache-dir terratorch
RUN pip install --no-cache-dir git+https://github.com/VMarsocci/pangaea-bench.git@main

HEALTHCHECK NONE
ENTRYPOINT []
CMD ["/bin/bash"]
