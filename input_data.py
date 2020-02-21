import os
import numpy as np
import random
import math


class DataSet:
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

    max_seq_length = 30

    def __gen_still(self, n):
        print("generating still synthetic pose sequence:")
        for i in range(n):
            pose = np.array([random.uniform(-0.2, 0.2), random.uniform(-0.25, 0.25), random.uniform(3.0, 3.2)])
            action_seq = []
            for f in range(DataSet.max_seq_length):
                action_seq.append(pose + np.array([random.uniform(-0.01, 0.01),
                                                   random.uniform(-0.01, 0.01),
                                                   random.uniform(-0.005, 0.005)]))
            self.add_x(action_seq)
            print(action_seq)
            self.add_y([1, 0, 0])

    def __gen_nod(self, n):
        print("generating nod synthetic pose sequence:")
        for i in range(n):
            pose = np.array([random.uniform(-0.2, 0.2), random.uniform(-0.25, 0.25), random.uniform(3.0, 3.6)])
            top = random.uniform(3.55, 3.65)
            bottom = random.uniform(2.95, 3.05)
            current_pose = [p for p in pose]
            is_downward = True
            action_seq = []
            for f in range(DataSet.max_seq_length):
                step = -random.uniform(0.002, 0.2) if is_downward else random.uniform(0.002, 0.2)
                current_pose = current_pose + np.array([random.uniform(-0.02, 0.02),
                                                        random.uniform(-0.02, 0.02),
                                                        step])
                if is_downward and current_pose[2] < bottom:
                    is_downward = False
                elif not is_downward and current_pose[2] > top:
                    is_downward = True
                action_seq.append(current_pose)
            print(action_seq)
            self.add_x(action_seq)
            self.add_y([0, 1, 0])

    def __gen_shake(self, n):
        print("generating shake synthetic pose sequence:")
        for i in range(n):
            pose = np.array([random.uniform(-0.2, 0.2), random.uniform(-0.6, 0.6), random.uniform(3.0, 3.3)])
            left = random.uniform(-0.65, -0.55)
            right = random.uniform(0.55, 0.65)
            current_pose = [p for p in pose]
            is_leftward = True
            action_seq = []
            for f in range(DataSet.max_seq_length):
                step = -random.uniform(0.01, 0.4) if is_leftward else random.uniform(0.01, 0.4)
                current_pose = current_pose + np.array([random.uniform(-0.02, 0.02),
                                                        step,
                                                        random.uniform(-0.1, 0.1)])
                if is_leftward and current_pose[1] < left:
                    is_leftward = False
                elif not is_leftward and current_pose[1] > right:
                    is_leftward = True
                action_seq.append(current_pose)
            print(action_seq)
            self.add_x(action_seq)
            self.add_y([0, 0, 1])

    def gen_syn_data(self, n_still, n_nod, n_shake):
        self.__gen_still(n_still)
        self.__gen_nod(n_nod)
        self.__gen_shake(n_shake)

    def train_test_split(self, ratio=0.33, shuffle=True):
        X = np.array([i for i in self.get_x()])
        Y = np.array([i for i in self.get_y()])
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
    data_set = DataSet()
    if not os.path.exists(filepath):
        print('Warning: ' + filepath + ' does not exist')
        return data_set

    for path in os.listdir(filepath):
        print("     ", path, " type: ", end="")
        # Use filename to determine action type
        #   1) still 0
        #   2) nod   1
        #   3) shake 2
        if path.find("nod") != -1:
            y = [0, 1, 0]
            print("nod")
        elif path.find("shake") != -1:
            y = [0, 0, 1]
            print("shake")
        else:
            y = [1, 0, 0]
            print("still")

        file = open(os.path.join(filepath, path))
        lines = file.readlines()
        begin_idx = 0
        while True:
            end_idx = min(begin_idx + DataSet.max_seq_length, lines.__len__())
            print(begin_idx, end_idx)
            action_seq = []
            for i in range(begin_idx, end_idx):
                words = lines[i].split()
                if words.__len__() != 6:
                    print(words.__len__())
                    continue
                action_seq.append([float(words[0]),
                                   float(words[1]),
                                   float(words[2]) if float(words[2]) > 0 else (float(words[2]) + math.pi * 2)])
            for i in range(end_idx, DataSet.max_seq_length):
                action_seq.append([0, 0, 0])
            if end_idx == lines.__len__():
                break
            else:
                begin_idx = end_idx
            data_set.add_x(action_seq)
            data_set.add_y(y)
        # if lines.__len__() < DataSets.max_seq_length:
        #     continue
        # for end_idx in range(DataSets.max_seq_length, lines.__len__()):
        #     action_seq = []
        #     begin_idx = max(end_idx - max_seq_length, 0)
        #     for i in range(begin_idx, end_idx):
        #         words = lines[i].split()
        #         if words.__len__() != 6:
        #             continue
        #         action_seq.append([float(words[0]),
        #                            float(words[1]),
        #                            float(words[2]) if float(words[2]) > 0 else (float(words[2]) + math.pi * 2)])
        #     data_set.add_x(action_seq)
        #     data_set.add_y(y)

    return data_set
