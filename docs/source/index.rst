ILAB Pangaea Repo
=================

Welcome to the ILAB fork of the Pangaea Foundation Model benchmarking repo. 
We have developed this repo to allow for easy use of Earth Observation Foundation Models at NASA Goddard.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   config_requirements
   troubleshooting

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Getting Started
---------------

To get started, clone the repository at `https://github.com/nasa-nccs-hpda/ilab-pangaea-bench.git <https://github.com/nasa-nccs-hpda/ilab-pangaea-bench.git>`_. 
This repository is split into 2 main parts, configs and pangaea. These represent two sides of the code: pangaea hosts the python code that is executed, config contains files that edit the functionality of the pangaea code. For example, a custom dataset must be coded in the pangaea dataset folder, and must be configured in the configs dataset folder. 

For most applications, this means that very little coding must be done, since much of what's added to this repo supports many different combinations of user configuration. Thus, most of users' time will be spent configuring behaviors in the configs directory. 

Adding custom classes
~~~~~~~~~~~~~~~~~~~~~

The first part to adding functionality to the repo is to add python classes that represent your custom datasets, 
model encoders/decoders, preprocessing (found under `engine/data_preprocessor.py`), 
and other custom training behaviors like loss functions, lr scheduling, and optimizers (found under `utils`). 

**For most users**: Creating a custom dataset python class is all of the coding you will need if you want to load your own data.
If you wish to use an already-supported dataset, see the subsection below this about config files. 

Adding config files
~~~~~~~~~~~~~~~~~~~

The second part is to configure these classes to integrate properly with the existing repo functionality. This is done by 
creating and editing corresponding .yaml files. For example, if you create a Dataset class called MyDataset, you will need to
make a .yaml file that outlines some of the dataset's functionality. This file will need to have a field called \_target\_ that
references MyDataset, which lets the driver script know where to look when you build your dataset. This behavior is true for
any modifications you make in Pangaea, so all preprocessing, loss functions, etc will also require a corresponding config file. 

For a full list of all of the required fields in a given config file, see **`THIS PAGE` (TODO: add this)**

If you are working with an existing class but wish to edit its functionality (like an encoder for a foundation model), you
should also create a config that mimics the existing config file for that class. For example, you may wish to run Terramind, 
but with a different amount of modalities and bands than the default configuration. Duplicating and editing the file is 
required to achieve this functionality. 

More information
~~~~~~~~~~~~~~~~
Below are some links to files that are part of the official Pangaea repo, which contain some more in-depth information.

* `Contributing Guide <CONTRIBUTING.md>`_: contains more information on the steps necessary to get a model up and running.
* `Dataset Guide <DATASET_GUIDE.md>`_: contains more information on implementing custom datasets, and some examples of running some already-implemented datasets.

Training
--------

Training a model is done using a console command, which we've simplified in our training notebook (see the `notebooks` folder for this and other examples). 

To train a model, you must ensure that you've modified the appropriate .yaml files, as mentioned above. Some functionality
(such as loading a user-created dataset) will also require coding a custom python class. Some require their own file (such as a
custom encoder, decoder, or dataset), while others need to be added to existing files (preprocessing needs to be added to
`data_preprocessor.py`, for example). See the file structure and look within existing files for additional information. 

Here is a checklist of .yaml files that need to be edited (and corresponding python classes if additional custom functionality
is desired):

* `dataset`: Information of downstream datasets such as image size, band_statistics, classes etc. 
* `decoder`: Downstream task decoder fine-tuning related parameters, like the type of architecture (e.g. UPerNet), which multi-temporal strategy to use, and other related hparams (e.g. nr of channels)
* `encoder`: GFM encoder related parameters. `output_layers` is used for which layers are used for Upernet decoder.  
* `preprocessing`: Both preprocessing and augmentations steps required for the dataset, such as bands adaptation, normalization, resize/crop.
* `task`: Information about the trainer and evaluator. Most of the parameters are overwritten in run. Trainer and evaluator can be used for segmentation (`SegTrainer`) or regression (`RegTrainer`). Different parameter like precision training (`precision`) can be set in it.
* `train`: This controls the global settings for training. The `finetune` parameter allows for the encoder to be trained alongside the decoder (by default the encoder is "frozen"). Some other PyTorch functionality, such as number of workers and batch size, can also be edited here. 
