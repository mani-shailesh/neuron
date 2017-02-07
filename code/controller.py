import json
import matplotlib.pyplot as plt

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


def create_model(input_shape, weight_decay, hidden_units=None, loss='CrossEntropy'):
    model_name = 'model_sigmoid_unnormalized_test'
    if hidden_units is not None:
        log_file = '../results/mlp_sigmoid_unnormalized_test.csv'
        dense1 = layers.Dense(hidden_units, weight_decay=weight_decay, input_shape=input_shape, name="dense_1")
        act1 = layers.Sigmoid(input_layer=dense1, name="sigmoid_1")
        dense2 = layers.Dense(10, weight_decay=weight_decay, input_layer=act1, name="dense_2")
        act = layers.Softmax(input_layer=dense2, name="softmax_1")
    else:
        log_file = '../results/mlp.csv'
        dense1 = layers.Dense(10, weight_decay=weight_decay, input_shape=input_shape, name="dense_1")
        act = layers.Softmax(input_layer=dense1, name="softmax_1")
    return models.MLP(input_layer=dense1, output_layer=act, loss=loss,
                      log_file=log_file, name=model_name)


def run_mlp_experiments():
    """
    Run basic experiments on the multi-layer perceptron
    """
    data_store = data_processor.DataStore()
    data_dict = data_store.get_data(normalize=False, split_val=False)

    train_X = data_dict['train']['data']
    train_Y = data_dict['train']['labels']

    # val_X = data_dict['val']['data']
    # val_Y = data_dict['val']['labels']

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

    mlp = create_model(input_shape, weight_decay, hidden_units, loss='CrossEntropy')

    train_acc, val_acc, epoch = \
        mlp.train(train_X, train_Y, lr, batch_size, num_epochs, None, None,
                  print_acc=False, save_dir=None)
    # util.log(log_filename,
    #          str(hidden_units) + "," + str(epoch) + "," + str(train_acc) +
    #          "," + str(val_acc))

    print("\nBest Validation Accuracy: " + str(val_acc) + ", Training accuracy: " + str(train_acc))
    test_acc = util.get_accuracy(test_Y, mlp.predict_classes(test_X))
    print("Test Accuracy: " + str(test_acc))


def create_model_from_dict(config_dict, input_shape):
    model_dict = config_dict['model']
    if model_dict['type'].lower() == 'mlp':
        input_layer = None
        output_layer = None
        for idx in range(model_dict['num_hidden_layers']):
            if output_layer is None:
                dense = layers.Dense(
                    num_units=model_dict['num_hidden_units'][idx],
                    weight_decay=config_dict['weight_decay'],
                    input_shape=input_shape,
                    name="dense"+str(idx+1)
                )
                input_layer = dense
            else:
                dense = layers.Dense(
                    num_units=model_dict['num_hidden_units'][idx],
                    weight_decay=config_dict['weight_decay'],
                    input_layer=output_layer,
                    name="dense" + str(idx + 1)
                )
            act_class = getattr(layers, model_dict['activation'])
            act = act_class(
                input_layer=dense,
                name=model_dict['activation']+str(idx)
            )
            output_layer = act
        if output_layer is None:
            dense_final = layers.Dense(
                num_units=10,
                weight_decay=config_dict['weight_decay'],
                input_shape=input_shape,
                name="dense_final"
            )
            input_layer = dense_final
        else:
            dense_final = layers.Dense(
                num_units=10,
                weight_decay=config_dict['weight_decay'],
                input_layer=output_layer,
                name="dense_final"
            )
        act_final = layers.Softmax(
            input_layer=dense_final,
            name="softmax"
        )
        output_layer = act_final
        model = models.MLP(
            input_layer=input_layer,
            output_layer=output_layer,
            loss=str(model_dict['loss']),
            log_file=config_dict['log_file'],
            name=model_dict['name']
        )
    elif model_dict['type'].lower() == 'rmlr':
        model = models.RMLR(
            num_classes=10,
            weight_decay=config_dict['weight_decay'],
            name=model_dict['name'],
            log_file=config_dict['log_file']
        )
    else:
        print("No model of type " + model_dict['type'] + "found!")
        model = None
    return model


def main(config_json_file):
    with open(config_json_file, 'r') as config_json_obj:
        config_dict = json.load(config_json_obj)

    data_store = data_processor.DataStore()
    data_dict = data_store.get_data(normalize=config_dict['normalize'])

    train_x = data_dict['train']['data']
    train_y = data_dict['train']['labels']

    val_x = data_dict['val']['data']
    val_y = data_dict['val']['labels']

    # test_x = data_dict['test']['data']
    # test_y = data_dict['test']['labels']

    input_shape = (config_dict['batch_size'], train_x.shape[1])

    print("\nCreating model...")
    model = create_model_from_dict(config_dict, input_shape)
    print("Done.")

    if model is not None:
        print("\nTraining model...")
        train_acc_list, val_acc_list = model.train(
            X=train_x,
            Y=train_y,
            lr=config_dict['learning_rate'],
            batch_size=config_dict['batch_size'],
            num_epochs=config_dict['num_epochs'],
            val_X=val_x,
            val_Y=val_y,
            print_acc=config_dict['print_acc'],
            save_dir=config_dict['save_dir']
        )
        print("Done.")

        max_val_acc = max(val_acc_list)
        epoch_idx = val_acc_list.index(max_val_acc)
        print("Maximum validation accuracy is " +
              str(max_val_acc) +
              " at epoch #" +
              str(epoch_idx + 1) +
              ". Training accuracy at that epoch is " +
              str(train_acc_list[epoch_idx]))

        if config_dict['show_plot']:
            plot_train, = plt.plot(
                range(1, config_dict['num_epochs'] + 1),
                train_acc_list,
                label="Training Accuracy"
            )
            plot_val, = plt.plot(
                range(1, config_dict['num_epochs'] + 1),
                val_acc_list,
                label="Validation Accuracy"
            )
            plot_val.axes.set_xlabel('# Epochs')
            plot_val.axes.set_ylabel('% Accuracy')
            plt.legend(ncol=1, fancybox=True, shadow=True)
            plt.show(block=False)
            print("Plots have been displayed.")
