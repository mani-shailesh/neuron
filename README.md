# neuron
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/mani-shailesh/neuron/blob/master/LICENSE.md)
[![Category](https://img.shields.io/badge/Category-Coursework-ff69b4.svg)](https://github.com/mani-shailesh/neuron)

This is a very basic and easy to use machine learning (especially MLP) framework where you build your network by putting together few layers and then train it. You can save the network, load it later and train it further or use it for predictions. 

Getting Started:

To just get started and perform some training, follow these simple steps:

1.  Download and extract files from http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz in the `data` directory.
1.  cd into `code` directory
2.  Prepare your configuration file or use one from the provided samples in `code/sample_config`. More details on each option in this configuration file are available later in this readme file.
3.  run `python controller.py config_file_path` in the terminal. If you do not provide `config_file_path`, it uses 'config.json' in the working directory.

Configuration File:

Configuration file describes the model and hyper parameters to the script. It is a JSON file with following keys:

    "num_epochs"    -   Number of epochs to run,
    "batch_size"    -   Mini-batch size,
    "save_dir"  -   Path to directory where model is to be saved. Model is not saved if this is null,
    "log_file"  -   Path to file where log is to be written. Log is not written if this is null,
    "learning_rate" -   Learning rate,
    "weight_decay" -   Weight decay for L2 regularization,
    "normalize" -   Input data is normalized, divided by standard deviation, if this is true. Input is always zero-centered.
    "show_plot" -   A plot of training and validation accuracies vs. number of epochs is displayed if this is true,
    "print_acc" -   Accuracies after each epoch are printed if this is true.
    
    
    "model" -   This describes the model to be created. It is again a JSON object with certain mandatory keys and some optional keys depending on model type. Mandatory keys are the following:

        "type"  -   Can be either "RMLR" for logistic regression or "MLP" for multi-layer perceptron
        "name"  -   Name of the model. This is used for naming the files when saving the model
        
        Following keys are mandatory only if model type is "MLP":
        
        "num_hidden_layers" -   Number of hidden layers. Can take any integer >= 0. 0 means a single layer MLP equivalent to logistic regression,
        "num_hidden_units"  -   List of integers >0 and length equal to "num_hidden_layers" representing number of units in each layer,
        "activation"    -   Name of the activation function to be used. Can take any value in ["ReLU", "LeakyReLU", "Sigmoid", "Tanh"]. Note that this is case-sensitive,
        "loss"  -   Name of the loss function to be used. Can take any value in ["CrossEntropy", "Hinge"]. Note that this is case-sensitive.
        
Some sample configuration files have been provided in the `code/sample_config` directory. A configuration file with best-performing parameters is also present in the 'code' directory as the default file. It is named 'config.json'.


The `controller` script uses CIFAR-10 dataset by default which it reads on the go from the `data` directory. However, the framework in itself is independent of the data source. You can use any dataset by implementing your own `DataStore` class like the one in `data_processor.py`.
