ILAB Pangaea Repo
=================

Welcome to the ILAB fork of the Pangaea Foundation Model benchmarking repo. 
We have developed this repo to allow for easy use of Earth Observation Foundation Models at NASA Goddard. 

Getting Started
---------------

To get started, clone the repository at `https://github.com/nasa-nccs-hpda/ilab-pangaea-bench.git`. 
This repository is split into 2 main parts: configs and pangaea itself. 

Adding custom classes
~~~~~~~~~~~~~~~~~~~~~

The first part to adding functionality to the repo is to add python classes that represent your custom datasets, 
model encoders/decoders, preprocessing (found under `engine/data_preprocessor.py`), 
and other custom training behaviors like loss functions, lr scheduling, and optimizers (found under `utils`). 

Adding config files
~~~~~~~~~~~~~~~~~~~

The second part is to configure these classes to integrate properly with the existing repo functionality. This is done by 
creating and editing corresponding .yaml files. For example, if you create a Dataset class called MyDataset, you will need to
make a .yaml file that you will reference during training/testing. This file will need to have a field called \_target\_ that
references MyDataset, which lets the driver script know where to look when you build your dataset. This behavior is true for
any modifications you make in Pangaea, so all preprocessing, loss functions, etc will also require a corresponding config file.

If you are working with an existing class but wish to edit its functionality (like an encoder for a foundation model), you
should also create a config that mimics the existing config file for that class. For example, you may wish to run Terramind, 
but with a different amount of modalities and bands than the default configuration. Duplicating and editing the file is 
required to achieve this functionality. 
