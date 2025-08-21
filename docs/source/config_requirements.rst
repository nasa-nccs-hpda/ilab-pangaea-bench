Config File Requirements
========================

The structure of this repo dictates that code is always paired with a config file to edit its functionality. This page will help you understand which files need to be edited, and what they need to contain. Each config file is found under the main repository folder's `config` directory. Subfolders contain files for similar use cases, such as dataset or encoder config files. Each file is of type ``.yaml``.

Default fields (found in all config files)
------------------------------------------

* ``_target_``: contains the Python module syntax for the class to use. In other words, if a config has a target called ``torch.nn.MSELoss``, Python will look for the MSELoss class in the Python package ``torch`` (also known as PyTorch), under the module ``nn``. 
* Other parameters: If you code a custom class that needs certain parameters, these need to be present in the config. This is how the driver script (``run.py``) knows how to initialize your custom class. 

**Note**: Any fields that are listed in the subsections below are **in addition to** the fields listed in the bulletpoints above. 

Criterion
---------
Also known as loss functions. Contains only default fields (see "all configs" section).

Dataset
-------
Represents the data the model will use to train, validate, and test.

Example: ``sen1floods11.yaml``

* ``dataset_name``: The final part of the ``_target_``, also known as the class name. 
* ``root_path``: Where to gather data from. If data is downloaded, this must be the destination folder for the download. 
* ``download_url``: For downloaded data, enter the correct URL. Otherwise this can be left blank. 
* ``auto_download``: Whether to automatically download or prompt the user to download. For local data, this can be left ``False``. 
* ``img_size``: Size of each image file in pixels. Assumes image height and width are the same.
* ``multi_temporal``: Whether data is multi temporal, True or False. 
* ``multi_modal``: Whether data is multi modal, True or False. 
* ``ignore_index``: Indices to ignore when loading data. ``-1`` is the default value. 
* ``num_classes``: Number of classes for classification. Default value is ``1``, can be left alone for non-classification tasks. 
* ``classes``: List of class names.
* ``distribution``: Distribution of data among classes. For non-classification tasks this can be left blank or be filled with dummy values.
* ``bands``: Contains a list of bands, separated by modality. The default modality for single-modality data is optical; if you are unsure of your modality, just put band numbers as optical. Band numbers should be of the form: 
  
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

* (any other custom parameters for dataset class)

Decoder
-------

Represents the ML framework used to "decode" the embeddings learned by the Foundation Model encoder (see below).

Example: ``seg_upernet.yaml``

* ``encoder``: leave this as ``null``, as this will populate automatically. 
* ``num_classes``: leave this as ``${dataset.num_classes}``, as this will populate automatically. 
* ``finetune``: leave this as ``false``, as this will populate automatically. 
* (any other custom parameters for decoder class)

Encoder
-------

Represents the Foundation Model that's being used for the desired task.

Example: ``prithvi.yaml``

* ``encoder_weights``: relative path to encoder weights file. This should have the form: ``./pretrained_models/{weights_filename}``.
* ``download_url``: URL to download weights from.
* ``input_size``: size of image, in pixels. Assumes image height and width are the same.
* ``input_bands``: list of input band names by modality. Must be of the form: 

  .. code-block:: yaml
      
    input_bands:
      - optical:
        - B1
        - B2
        - (etc)

* ``output_layers``: index of output layers, must be a list. Example: 

  .. code-block:: yaml
      
    output_layers:
      - 3
      - 5
      - 7
      - 11

* ``output_dim``: size of output. Assumes output embedding height and width are the same. 
* (any other custom parameters for encoder class)

LR Scheduler
------------

Schedules the learning rate (LR) of the model during training. Contains only default fields (see "all configs" section).

Optimizer
---------

Also aids in learning rate adjustments during training. Contains only default fields (see "all configs" section).

Preprocessing
-------------

This defines what preprocessing will occur on the data during different stages of model use. 

Example: ``seg_default.yaml``

* ``train``, ``val``, ``test``: which type of processing to apply during each phase. These can all be the same or all be different depending on the application. Each section has its own subsection, as listed below. 
  
  * ``_target_``: always set to ``pangaea.engine.data_preprocessor.Preprocessor``, since all preprocessing is done by this class. 
  * ``preprocessor_cfg``: for specific preprocessor class that inherits from the base preprocessor. Contains multiple copies of ``_target_``, depending on how many transforms will happen sequentially. There must be at least 1 ``_target_`` present.
    
    * ``_target_``: usual syntax (see "all configs" section). Since all preprocessing happens in pangea.engine.data_preprocessor.py, this must be of the form ``pangaea.engine.data_preprocessor.{Class_Name}``.

Example: 

  .. code-block:: yaml
    
    train:
        _target_: pangaea.engine.data_preprocessor.Preprocessor
        preprocessor_cfg:
            - _target_: pangaea.engine.data_preprocessor.PBMinMaxNorm

Task
----

Represents the desired Machine Learning task being performed by the model (FM encoder, and decoder). This can be regression, segmentation, or any other function outlined by the ``.yaml`` files in the directory, or a custom task.

* ``trainer``: represents the PyTorch Lightning Trainer object used to train the model. This section has a llist of parameters that the trainer requires. 
  
  * ``_target_``: set to desired task-specific trainer (``pangaea.engine.trainer.{Trainer_ClassName}``). Can code a custom trainer if desired.
  * Parameters overwritten in ``run.py``: leave as the default value or hard-code here
    
    * ``model``: ``null``
    * ``train_loader``: ``null``
    * ``optimizer``: ``null``
    * ``lr_scheduler``: ``null``
    * ``evaluator``: ``null``
    * ``exp_dir``: ``null``
    * ``device``: ``null``
    * ``criterion``: ``null`` 
  
  * Parameters to adapt: 
    
    * ``n_epochs``: number of epochs to train for.
    * ``precision``: default value is ``fp32``, can be changed to a different value. Uses PyTorch literals for numerical formats (``int8``, ``fp64``, etc).
    * ``ckpt_interval``: how often to save a model checkpoint (save every ``ckpt_interval`` epochs).
    * ``eval_interval``: how often to run evaluation suite (eval every ``eval_interval`` epochs). Best to keep as the same value as ``log_interval`` for accuracy of metrics. 
    * ``log_interval``: how often to log statistics (log every ``eval_interval`` epochs). Best to keep as the same value as ``eval_interval`` for accuracy of metrics. 
    * ``best_metric_key``: which metric to use when determining the best model checkpoint. Uses PyTorch syntax (mIoU, val-loss, etc).
    * ``use_wandb``: ``${use_wandb}`` by default, can be hard-coded to ``true`` or ``false``. 

* ``evaluator``: represents the PyTorch Lightning Trainer object used to evaluate the model (every ``eval_interval`` epochs, as set in the trainer).
  
  * ``_target_``: set to desired task-specific evaluator (``pangaea.engine.trainer.{Evaluator_ClassName}``). Can code a custom evaluator if desired.
  * Parameters overwrittern in ``run.py``: leave as the default value or hard-code
    
    * ``val_loader``: ``null``
    * ``exp_dir``: ``null``
    * ``device``: ``null``
    * ``use_wandb``: ``${use_wandb}``
    * ``inference_mode``: ``null``
    * ``sliding_inference_batch``: ``null``

Train
-----------
This config defines the training behavior of the ``run.py`` script, using PyTorch Lightning. Make a copy or edit the train.yaml directly to change behavior.

Basic options: these can often be left alone.

* ``train``: leave as ``true``
* ``work_dir``: where to save model outputs (checkpoints, logs, etc). Empty string is default value, so this defaults to current working directory.
* ``seed``: random seed to use in PyTorch Lightning. ``234`` by default.
* ``use_wandb``: whether to use wandb for experiment tracking. ``false`` by default. See `documentation <https://wandb.ai/site/>`_ for reference.
* ``wandb_run_id``: what to name the wandb run. ``null`` by default.

Parallelization options: increase based on your hardware, larger numbers mean more parallelization but also more compute.

* ``num_workers``: how many PyTorch lightning workers to use. ``4`` by default.
* ``batch_size``: how many images per training batch to supply. ``8`` by default.
* ``test_num_workers``: how many workers to use for testing. ``4`` by default.
* ``test_batch_size``: how many images per testing batch to supply. ``1`` by default.

Hyperparameters and other options:

* ``finetune``: whether to finetune encoder weights. ``false`` by default (frozen encoder).
* ``ckpt_dir``: where to save model checkpoint
* ``limited_label_train``: ``1`` by default.
* ``limited_label_val``: ``1`` by default
* ``limited_label_strategy``: Pick from ``stratified, oversampled, random``. ``stratified`` by default. 
* ``stratification_bins``: number of stratification bins, ignore if not using stratified. ``3`` by default.
* ``data_replicate``: ``1`` by default.
* ``use_final_ckpt``: Whether to use final checkpoint for testing. ``false`` by default, so best checkpoint (according to metric defined in task ``.yaml`` file) will be used.

Defaults: keep these as they are, these will be overwritten in ``run.py`` during training.

.. code-block:: yaml

    defaults:
    - task: ???
    - dataset: ???
    - encoder: ???
    - decoder: ???
    - preprocessing: ???
    - criterion: ???
    - lr_scheduler: multi_step_lr
    - optimizer: adamw
    - _self_

Test
----

* ``train``: leave as ``true``
* ``work_dir``: where to save model outputs (checkpoints, logs, etc). Empty string is default value, so this defaults to current working directory.
* ``seed``: random seed to use in PyTorch Lightning. ``234`` by default.
* ``use_wandb``: whether to use wandb for experiment tracking. ``false`` by default. See `documentation <https://wandb.ai/site/>`_ for reference.
* ``wandb_run_id``: what to name the wandb run. ``null`` by default.
* ``num_workers``: how many PyTorch lightning workers to use. ``1`` by default.
* ``batch_size``: how many images per training batch to supply. ``8`` by default.
* ``use_final_ckpt``: Whether to use final checkpoint for testing. ``false`` by default, so best checkpoint (according to metric defined in task ``.yaml`` file) will be used.
* ``finetune``: ``false`` by default. 
* ``ckpt_dir``: ``???`` by default.






