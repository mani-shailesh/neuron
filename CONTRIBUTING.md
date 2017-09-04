Follwing is a brief description of the modules:

`data_processor.py`   -   This module deals with all the data related operations. It contains the definition of `DataStore` class responsible for reading the data and applying pre-processing operations on it. `DataStore.load_data()` can be used to get raw data loaded from the 'data' directory.

`layers.py` -   This module contains all the basic layers which are building blocks for a MLP model. All layers inherit from the base class `Layer`. This also contains the loss functions which inherit from the base class `Loss`.

`models.py` -   This module contains the two basic models supported by this platform which are `RMLR`, regularized multinomial logistic regression and `MLP`, multi layer perceptron. These classes expose basic and consistent methods like `train()` and `predict_classes()`.
 
 `util.py`  -   This module contains utility functions which are independent of any other module like `log()` and `get_accuracy()`.
 
 `controller.py`    -   This is the driver script which puts together layers or models as per the configuration provided by the user.
 
 All modules are thoroughly documented and you can always refer to the inline comments for more details.
