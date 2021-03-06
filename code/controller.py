#!/usr/bin/python
import sys
import json
import matplotlib.pyplot as plt
import numpy as np

import data_processor
import layers
import models
import util


def create_model_from_dict(config_dict, input_shape):
    """
    Create model from the configuration dictionary read from json file.
    :param config_dict: Configuration dictionary sufficient to define model
    :param input_shape: Shape of the input tensor
    :return: object of appropriate model
    """
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
                    name="dense" + str(idx + 1)
                )
                input_layer = dense
            else:
                dense = layers.Dense(
                    num_units=model_dict['num_hidden_units'][idx],
                    weight_decay=config_dict['weight_decay'],
                    input_layer=output_layer,
                    name="dense" + str(idx + 1)
                )
            act_class = getattr(layers, str(model_dict['activation']))
            act = act_class(
                input_layer=dense,
                name=str(model_dict['activation']) + str(idx + 1)
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
    """
    Main driver function to read configuration file, create corresponding model,
    and train it.
    :param config_json_file: Full path to the configuration json file
    :return:
    """
    with open(config_json_file, 'r') as config_json_obj:
        config_dict = json.load(config_json_obj)

    data_store = data_processor.DataStore()
    data_dict = data_store.get_data(normalize=config_dict['normalize'])

    train_x = data_dict['train']['data']
    train_y = data_dict['train']['labels']

    val_x = data_dict['val']['data']
    val_y = data_dict['val']['labels']

    test_x = data_dict['test']['data']
    test_y = data_dict['test']['labels']

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
            print_loss=config_dict['print_acc'],
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

        print("Calculating test accuracy...")
        test_acc = util.get_accuracy(test_y, model.predict_classes(test_x))
        print("Test accuracy after " +
              str(config_dict['num_epochs']) +
              " epochs is " +
              str(test_acc) + ".")

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
            plt.show(block=True)
            print("Plots have been displayed.")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        config_file = str(sys.argv[1])
    else:
        config_file = 'config.json'
    main(config_file)


def plot_sequences(sequence_list, label_list, title, xlabel, ylabel):
    for idx, sequence in enumerate(sequence_list):
        plot_obj, = plt.plot(
            range(1, len(sequence) + 1),
            sequence,
            label=label_list[idx]
        )
    plot_obj.axes.set_xlabel(xlabel)
    plot_obj.axes.set_ylabel(ylabel)
    plt.legend(ncol=1, fancybox=True, shadow=True)
    plt.title(title)
    plt.show(block=True)
