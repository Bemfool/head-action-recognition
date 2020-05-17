import os
import numpy as np
import random
import math
from sklearn import preprocessing


class FpDataSet:
    def __init__(self):
        self.__X = []
        self.__Y = []

    def add_x(self, x):
        self.__X.append(x)

    def add_y(self, y):
        self.__Y.append(y)

    def get_x(self):
        return np.array(self.__X)

    def get_y(self):
        return np.array(self.__Y)

    max_seq_length = 60

    def train_test_split(self, ratio=0.33, shuffle=True):
        X = np.array([i for i in self.get_x()])
        Y = np.array([i for i in self.get_y()])
        print(X.__len__())
        print(X[0].__len__())
        print(X[0][0].__len__())
        len = X.__len__()
        if shuffle:
            idx = np.arange(len)
            np.random.shuffle(idx)
            X = X[idx, :, :]
            Y = Y[idx, :]
        train_len = int(len * (1. - ratio))
        X_test = X[train_len:, :, :]
        X_train = X[:train_len, :, :]
        Y_test = Y[train_len:, :]
        Y_train = Y[:train_len, :]
        return X_train, Y_train, X_test, Y_test


def read_data_sets(filepath):
    data_set = FpDataSet()
    if not os.path.exists(filepath):
        print('Warning: ' + filepath + ' does not exist')
        return data_set

    for path in os.listdir(filepath):
        print("     ", path, " type: ", end="")
        # Use filename to determine action type
        #   1) still 0
        #   2) nod   1
        #   3) shake 2
        if path.find("-y") != -1:
            y = [1, 0, 0, 0, 0, 0]
            print("yaw -")
        elif path.find("+y") != -1:
            y = [0, 1, 0, 0, 0, 0]
            print("yaw +")
        elif path.find("-p") != -1:
            y = [0, 0, 1, 0, 0, 0]
            print("pitch -")
        elif path.find("+p") != -1:
            y = [0, 0, 0, 1, 0, 0]
            print("pitch +")
        elif path.find("-r") != -1:
            y = [0, 0, 0, 0, 1, 0]
            print("roll -")
        elif path.find("+r") != -1:
            y = [0, 0, 0, 0, 0, 1]
            print("roll +")
        else:
            y = [0, 0, 0, 0, 0, 0]
            print("ERROR: Unchecked file")

        file = open(os.path.join(filepath, path))

        def not_empty(l):
            return l and l.strip()

        lines = list(filter(not_empty, file.readlines()))
        action_seq = []
        for line in lines:
            words = line.split()

            coords = []
            for j in range(68):
                coords.append([float(words[j * 2]), float(words[j * 2 + 1])])

            min_max_scaler = preprocessing.MinMaxScaler()
            coords = min_max_scaler.fit_transform(coords)

            action_seq.append(coords.flatten())
        print(action_seq.__len__())
        for i in range(FpDataSet.max_seq_length - action_seq.__len__()):
            action_seq.append([0] * 136)

        print(action_seq.__len__())
        data_set.add_x(action_seq)
        data_set.add_y(y)

    return data_set
