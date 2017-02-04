import data_processor
import layers
import models
import util


def run_rmlr_experiments():
    """
    Run experiments on regularized logistic multinomial regressor (rmlr)
    """
    data_store = data_processor.DataStore()
    data_dict = data_store.get_data()

    batch_size_list = [10, 50, 100, 200, 400, 1000]
    lr_list = [pow(10, -1 * ii) for ii in range(1, 9)]
    # batch_size = 100
    # lr = 1e-6

    num_epochs = 500
    # lambda_list = [pow(2, -1 * ii) for ii in range(1, 10)] + [0]
    lambda_ = 0.04

    log_filename = '../results/rmlr_batch_lr.csv'
    util.log(log_filename, 'batch,lr,epoch#,train_acc,val_acc')

    train_X = data_dict['train']['data']
    train_Y = data_dict['train']['labels']

    val_X = data_dict['val']['data']
    val_Y = data_dict['val']['labels']

    for lr in lr_list:
        for batch_size in batch_size_list:
            print("\n---------------------------------------------------------\n")
            print("Learning Rate: " + str(lr) + ", Batch Size: " + str(batch_size)
                  + ", Weight Decay: " + str(lambda_) + "\n")

            rmlr = models.RMLR(10, '../results/rmlr.csv')
            train_acc, val_acc, epoch = rmlr.train(train_X, train_Y, lr, batch_size, num_epochs, lambda_,
                                                   val_X, val_Y, reinit_weights=True, print_acc=False)
            util.log(log_filename,
                     str(batch_size) + "," + str(lr) + "," + str(epoch)
                     + "," + str(train_acc) + "," + str(val_acc))

            print("\nBest Validation Accuracy: " + str(val_acc) + ", Training accuracy: " + str(train_acc))


def create_model(input_shape, weight_decay, hidden_units=None, loss=layers.CrossEntropy()):
    if hidden_units is not None:
        log_file = '../results/mlp_tanh.csv'
        dense1 = layers.Dense(hidden_units, weight_decay=weight_decay, input_shape=input_shape, name="dense_1")
        act1 = layers.Tanh(input_layer=dense1, name="tanh_1")
        dense2 = layers.Dense(10, weight_decay=weight_decay, input_layer=act1, name="dense_2")
        act = layers.Softmax(input_layer=dense2, name="softmax_1")
    else:
        log_file = '../results/mlp.csv'
        dense1 = layers.Dense(10, weight_decay=weight_decay, input_shape=input_shape, name="dense_1")
        act = layers.Softmax(input_layer=dense1, name="softmax_1")
    return models.MLP(input_layer=dense1, output_layer=act, loss=loss,
                      log_file=log_file)


def run_mlp_experiments():
    """
    Run basic experiments on the multi-layer perceptron
    """
    data_store = data_processor.DataStore()
    data_dict = data_store.get_data()

    train_X = data_dict['train']['data']
    train_Y = data_dict['train']['labels']

    val_X = data_dict['val']['data']
    val_Y = data_dict['val']['labels']

    test_X = data_dict['test']['data']
    test_Y = data_dict['test']['labels']

    # log_filename = '../results/mlp_relu_units.csv'
    # util.log(log_filename, 'units,epoch#,train_acc,val_acc')

    batch_size = 100
    num_epochs = 500

    lr = 0.001
    weight_decay = 0.03

    # hidden_units_list = [20, 50, 100, 250, 500, 1000, 2000]
    hidden_units = 500

    input_shape = (batch_size, train_X.shape[1])

    print("\n---------------------------------------------------------")
    print("Hidden units: " + str(hidden_units) + "\n")

    mlp = create_model(input_shape, weight_decay, hidden_units)

    train_acc, val_acc, epoch = \
        mlp.train(train_X, train_Y, lr, batch_size, num_epochs, val_X, val_Y, False)
    # util.log(log_filename,
    #          str(hidden_units) + "," + str(epoch) + "," + str(train_acc) +
    #          "," + str(val_acc))

    print("\nBest Validation Accuracy: " + str(val_acc) + ", Training accuracy: " + str(train_acc))
    test_acc = util.get_accuracy(test_Y, mlp.predict_classes(test_X))
    print("Test Accuracy: " + str(test_acc))
