# aml-lab-1
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/mani-shailesh/aml-lab-1/blob/master/LICENSE.md)
[![Category](https://img.shields.io/badge/Category-Coursework-ff69b4.svg)](https://github.com/mani-shailesh/aml-lab-1)

This is a very basic and easy to use machine learning (especially MLP) framework.

To just get started and perform some training, follow these simple steps:

1.  cd into 'code' directory
2.  Prepare your configuration file or use one from the provided samples. More details on each option are available later in this readme file.
3.  run 'python controller.py config_file_path' in the terminal. If you do not provide 'config_file_path', it uses 'config.json' in the working directory.

The 'controller' script uses CIFAR-10 dataset by default which it reads on the go from the 'data' directory. However, the framework in itself is independent of the data source. You can use any dataset by implementing your own 'DataStore' class like the one in 'data_processor.py'.  
