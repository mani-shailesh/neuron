from tqdm import tqdm

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

    batch_size_list = [10, 50, 100, 200, 400, 800]
    lr_list = [pow(10, -1 * ii) for ii in range(6, 10)]

    num_epochs = 600
    lambda_ = 0.001

    log_filename = '../results/rmlr_batch_lr.csv'
    util.log(log_filename, 'lr,batch_size,epoch#,train_acc,val_acc')

    rmlr = models.RMLR(10, '../results/rmlr.csv')

    train_X = data_dict['train']['data']
    train_Y = data_dict['train']['labels']

    val_X = data_dict['val']['data']
    val_Y = data_dict['val']['labels']

    for lr in tqdm(lr_list):
        for batch_size in tqdm(batch_size_list):
            print("Learning Rate: " + str(lr) + ", Batch Size: " + str(batch_size))
            print("\n---------------------------------------------------------\n")
            train_acc, val_acc, epoch = rmlr.train(train_X, train_Y, lr, batch_size, num_epochs, lambda_,
                                                   val_X, val_Y, True)
            util.log(log_filename,
                     str(lr) + "," + str(batch_size) + "," + str(epoch) + "," + str(train_acc) + "," + str(val_acc))


def create_model(input_shape, weight_decay):
    dense1 = layers.Dense(10, input_shape=input_shape, name="dense_1", weight_decay=weight_decay)
    act1 = layers.Softmax(input_layer=dense1, name="softmax_1")
    # dense2 = layers.Dense(10, input_layer=act1, name="dense_2")
    # act2 = layers.Softmax(input_layer=dense2, name="softmax_2")
    return models.MLP(input_layer=dense1, output_layer=act1, loss=layers.CrossEntropy(),
                      log_file='../results/mlp.csv')


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

    batch_size = 200
    num_epochs = 5
    lr = 0.001
    weight_decay = 0.001

    input_shape = (batch_size, train_X.shape[1])
    mlp = create_model(input_shape, weight_decay)

    mlp.train(train_X, train_Y, lr, batch_size, val_X, val_Y, num_epochs)
