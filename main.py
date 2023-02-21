"""
[Pytorch migration version, by twkim]

This code is a modified version for using pytorch, which is based on the code by Yarin Gal

# This file contains code to train dropout networks on the UCI datasets using the following algorithm:
# 1. Create 20 random splits of the training-test dataset.
# 2. For each split:
# 3.   Create a validation (val) set taking 20% of the training set.
# 4.   Get best hyperparameters: dropout_rate and tau by training on (train-val) set and testing on val set.
# 5.   Train a network on the entire training set with the best pair of hyperparameters.
# 6.   Get the performance (MC RMSE and log-likelihood) on the test set.
# 7. Report the averaged performance (Monte Carlo RMSE and log-likelihood) on all 20 splits.
"""

import torch

import math
import numpy as np
import argparse
import sys
import os

from subprocess import call
from net.net_torch import DropNet


def train(args):
    dataset_root = args.dataset
    data_directory = args.dir
    epochs_multiplier = args.epochx
    num_hidden_layers = args.hidden
    assert data_directory in os.listdir(dataset_root)

    print(args)

    _RESULTS_VALIDATION_LL = "./UCI_Datasets/" + data_directory + "/results/validation_ll_" + str(
        epochs_multiplier) + "_xepochs_" + str(num_hidden_layers) + "_hidden_layers.txt"
    _RESULTS_VALIDATION_RMSE = "./UCI_Datasets/" + data_directory + "/results/validation_rmse_" + str(
        epochs_multiplier) + "_xepochs_" + str(num_hidden_layers) + "_hidden_layers.txt"
    _RESULTS_VALIDATION_MC_RMSE = "./UCI_Datasets/" + data_directory + "/results/validation_MC_rmse_" + str(
        epochs_multiplier) + "_xepochs_" + str(num_hidden_layers) + "_hidden_layers.txt"

    _RESULTS_TEST_LL = "./UCI_Datasets/" + data_directory + "/results/test_ll_" + str(
        epochs_multiplier) + "_xepochs_" + str(num_hidden_layers) + "_hidden_layers.txt"
    _RESULTS_TEST_TAU = "./UCI_Datasets/" + data_directory + "/results/test_tau_" + str(
        epochs_multiplier) + "_xepochs_" + str(num_hidden_layers) + "_hidden_layers.txt"
    _RESULTS_TEST_RMSE = "./UCI_Datasets/" + data_directory + "/results/test_rmse_" + str(
        epochs_multiplier) + "_xepochs_" + str(num_hidden_layers) + "_hidden_layers.txt"
    _RESULTS_TEST_MC_RMSE = "./UCI_Datasets/" + data_directory + "/results/test_MC_rmse_" + str(
        epochs_multiplier) + "_xepochs_" + str(num_hidden_layers) + "_hidden_layers.txt"
    _RESULTS_TEST_LOG = "./UCI_Datasets/" + data_directory + "/results/log_" + str(
        epochs_multiplier) + "_xepochs_" + str(num_hidden_layers) + "_hidden_layers.txt"

    _DATA_DIRECTORY_PATH = "./UCI_Datasets/" + data_directory + "/data/"
    _DROPOUT_RATES_FILE = _DATA_DIRECTORY_PATH + "dropout_rates.txt"
    _TAU_VALUES_FILE = _DATA_DIRECTORY_PATH + "tau_values.txt"
    _DATA_FILE = _DATA_DIRECTORY_PATH + "data.txt"
    _HIDDEN_UNITS_FILE = _DATA_DIRECTORY_PATH + "n_hidden.txt"
    _EPOCHS_FILE = _DATA_DIRECTORY_PATH + "n_epochs.txt"
    _INDEX_FEATURES_FILE = _DATA_DIRECTORY_PATH + "index_features.txt"
    _INDEX_TARGET_FILE = _DATA_DIRECTORY_PATH + "index_target.txt"
    _N_SPLITS_FILE = _DATA_DIRECTORY_PATH + "n_splits.txt"

    def _get_index_train_test_path(split_num, train=True):
        """
           Method to generate the path containing the training/test split for the given
           split number (generally from 1 to 20).
           @param split_num      Split number for which the data has to be generated
           @param train          Is true if the data is training data. Else false.
           @return path          Path of the file containing the requried data
        """
        if train:
            return _DATA_DIRECTORY_PATH + "index_train_" + str(split_num) + ".txt"
        else:
            return _DATA_DIRECTORY_PATH + "index_test_" + str(split_num) + ".txt"

    print("Removing existing result files...")
    call(["rm", _RESULTS_VALIDATION_LL])
    call(["rm", _RESULTS_VALIDATION_RMSE])
    call(["rm", _RESULTS_VALIDATION_MC_RMSE])
    call(["rm", _RESULTS_TEST_LL])
    call(["rm", _RESULTS_TEST_TAU])
    call(["rm", _RESULTS_TEST_RMSE])
    call(["rm", _RESULTS_TEST_MC_RMSE])
    call(["rm", _RESULTS_TEST_LOG])
    print("Result files removed.")

    np.random.seed(1)

    print("Loading data and other hyperparameters...")
    data = np.loadtxt(_DATA_FILE)
    n_hidden = np.loadtxt(_HIDDEN_UNITS_FILE).tolist()
    n_epochs = np.loadtxt(_EPOCHS_FILE).tolist()
    index_features = np.loadtxt(_INDEX_FEATURES_FILE)
    index_target = np.loadtxt(_INDEX_TARGET_FILE)

    X = data[:, [int(i) for i in index_features.tolist()]]
    y = data[:, int(index_target.tolist())]

    n_splits = np.loadtxt(_N_SPLITS_FILE)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    print("* Dataset shapes::")
    print("    Raw data shape: {}".format(data.shape))
    print("    X shape: {}, y shape: {}".format(X.shape, y.shape))
    print("* Params::")
    print("    n_hidden {}, n_epochs: {}, n_splits: {}".format(n_hidden, n_epochs, n_splits))
    print("    device: {}".format(device))
    print("Done... ")

    errors, MC_errors, lls = [], [], []
    for split in range(int(n_splits)):
        print('Loading file: ' + _get_index_train_test_path(split, train=True))
        print('Loading file: ' + _get_index_train_test_path(split, train=False))
        index_train = np.loadtxt(_get_index_train_test_path(split, train=True))
        index_test = np.loadtxt(_get_index_train_test_path(split, train=False))
        print("index_train {}, index_test: {}".format(index_train.shape, index_test.shape))

        X_train = X[[int(i) for i in index_train.tolist()]]
        y_train = y[[int(i) for i in index_train.tolist()]]

        X_test = X[[int(i) for i in index_test.tolist()]]
        y_test = y[[int(i) for i in index_test.tolist()]]

        X_train_original = X_train
        y_train_original = y_train
        num_training_examples = int(0.8 * X_train.shape[0])
        X_validation = X_train[num_training_examples:, :]
        y_validation = y_train[num_training_examples:]
        X_train = X_train[0:num_training_examples, :]
        y_train = y_train[0:num_training_examples]

        # Printing the size of the training, validation and test sets
        print('Number of training examples: ' + str(X_train.shape[0]))
        print('Number of validation examples: ' + str(X_validation.shape[0]))
        print('Number of test examples: ' + str(X_test.shape[0]))
        print('Number of train_original examples: ' + str(X_train_original.shape[0]))

        # List of hyperparameters which we will try out using grid-search
        dropout_rates = np.loadtxt(_DROPOUT_RATES_FILE).tolist()
        tau_values = np.loadtxt(_TAU_VALUES_FILE).tolist()

        # We perform grid-search to select the best hyperparameters based on the highest log-likelihood value
        best_network = None
        best_ll = -float('inf')
        best_tau = 0
        best_dropout = 0
        for dropout_rate in dropout_rates:
            for tau in tau_values:
                print('Grid search step: Tau: ' + str(tau) + ' Dropout rate: ' + str(dropout_rate))
                # model = DropNet(X_train=)
                # network = net.net(X_train, y_train, ([int(n_hidden)] * num_hidden_layers),
                #                   normalize=True, n_epochs=int(n_epochs * epochs_multiplier), tau=tau,
                #                   dropout=dropout_rate)


if __name__ == '__main__':
    print("Dropout Uncertainty Experiments Pytorch migration!")
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', '-ds', default='UCI_Datasets', help='Root directory of dataset')
    parser.add_argument('--dir', '-d', default='bostonHousing', help='Name of the UCI Dataset directory. Eg: bostonHousing')
    parser.add_argument('--epochx', '-e', default=100, type=int, help='Multiplier for the number of epochs for training.')
    parser.add_argument('--hidden', '-nh', default=1, type=int, help='Number of hidden layers for the neural net')

    args = parser.parse_args()

    sys.path.append('net/')

    train(args=args)
