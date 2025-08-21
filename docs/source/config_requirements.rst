Config File Requirements
========================

The structure of this repo dictates that code is always paired with a config file to edit its functionality. This page will help you understand which files need to be edited, and what they need to contain. Each config file is found under the main repository folder's `config` directory. Subfolders contain files for similar use cases, such as dataset or encoder config files. Each file is of type ``.yaml``.

Default fields (found in all config files)
-----------

* ``\_target\_``: contains the Python module syntax for the class to use. In other words, if a config has a target called ``torch.nn.MSELoss``, Python will look for the MSELoss class in the Python package ``torch`` (also known as PyTorch), under the module ``nn``. 
* Other parameters: If you code a custom class that needs certain parameters, these need to be present in the config. This is how the driver script (``run.py``) knows how to initialize your custom class. 

**Note**: Any fields that are listed in the subsections below are **in addition to** the fields listed in the bulletpoints above. 

Criterion
-----------
Also known as loss functions. Contains only default fields (see "all configs" section).

Dataset
-------
Example: ``sen1floods11.yaml``.

* ``dataset_name``: The final part of the ``\_target\_``, also known as the class name. 
* ``root_path``: Where to gather data from. If data is downloaded, this must be the destination folder for the download. 
* ``download_url``: For downloaded data, enter the correct URL. Otherwise this can be left blank. 
* ``auto_download``: Whether to automatically download or prompt the user to download. For local data, this can be left False. 
* ``img_size``: Size of each image file in pixels. Images are expected to be square. 
* ``multi_temporal``: Whether data is multi temporal, True or False. 
* ``multi_modal``: Whether data is multi modal, True or False. 
* ``ignore_index``: Indices to ignore when loading data. -1 is the default value. 
* ``num_classes``: Number of classes for classification. Leave as 1 for other tasks. 
* ``classes``: Names of classes. Must be supplied as a list, using either tabbed dash characters, or pythonic [] syntax.
* ``distribution``: Distribution of data among classes. For non-classification tasks this can be left blank or be filled with dummy values.
* ``bands``: Contains a list of bands, separated by modality. The default modality for single-modality data is optical; if you are unsure of your modality, just put band numbers here. Band numbers should be input in the form: 
  
  .. code-block:: yaml
      
    bands:
      - optical:
        - B1
        - B2
        - (etc)
  
* ``data_mean``, ``data_std``, ``data_max``, ``data_min``: Statistics of the dataset, which are used during preprocessing. If your dataset does not use preprocessing, these can be filled with dummy values. Include one value per band, per modality.
  
  .. code-block:: yaml 
      
    data_mean:
      - optical:
        - 0.25
        - 0.1
        - (etc)

Train, Test
-----------
N/A



