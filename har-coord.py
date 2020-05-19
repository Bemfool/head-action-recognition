import tensorflow as tf
from keras import backend as K
from keras.models import load_model
import pandas as pd
import numpy as np
from keras.layers import Masking
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers.core import Dense, Dropout
import input_data_fp
import argparse
np.random.seed(42)
tf.set_random_seed(42)

sess = tf.Session(graph=tf.get_default_graph())
K.set_session(sess)

epochs = 200
batch_size = 64
n_hidden = 256
n_classes = 6

ACTIONS = {
    0: 'YAW_N',
    1: 'YAW_P',
    2: 'PITCH_N',
    3: 'PITCH_P',
    4: 'ROLL_N',
    5: 'ROLL_P',
}


def confusion_matrix(Y_true, Y_pred, tag_test):
    for i in range(Y_true.__len__()):
        t = np.argmax(Y_true[i])
        p = np.argmax(Y_pred[i])
        if t != p:
            print("True: ", str(t), " Pred: ", str(p), " - Tag: ", str(tag_test[i]), " *****")
        else:
            print("True: ", str(t), " Pred: ", str(p), " - Tag: ", str(tag_test[i]))

    Y_true = pd.Series([ACTIONS[y] for y in np.argmax(Y_true, axis=1)])
    Y_pred = pd.Series([ACTIONS[y] for y in np.argmax(Y_pred, axis=1)])

    return pd.crosstab(Y_true, Y_pred, rownames=['True'], colnames=['Pred'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-t',
        '--is_training',
        action="store_true",
        help='Set training mode'
    )
    parser.add_argument(
        '-m',
        '--model_filename',
        default='model.h5',
        help='Filename of the model to be loaded'
    )
    parser.add_argument(
        '-d',
        '--data_dir',
        default='./data',
        help='Directory of data'
    )

    parser.add_argument(
        '-c',
        '--is_continue',
        action="store_true",
        help="Load model and train"
    )

    args = parser.parse_args()
    data_set = input_data_fp.read_data_sets(args.data_dir)

    print(str(data_set.get_x().__len__()))
    print(str(data_set.get_x()[0].__len__()))
    print(str(data_set.get_x()[0][0].__len__()))
    print(str(data_set.get_y().__len__()))

    X_train, Y_train, X_test, Y_test, tag_test = data_set.train_test_split()
    if args.is_training:

        timesteps = len(X_train[0])
        input_dim = len(X_train[0][0])

        if args.is_continue:
            model = load_model(args.model_filename)
        else:
            model = Sequential()
            model.add(Masking(mask_value=[0]*input_dim, input_shape=(timesteps, input_dim)))
            model.add(LSTM(n_hidden, input_shape=(timesteps, input_dim)))
            model.add(Dropout(0.5))
            model.add(Dense(n_classes, activation='softmax'))

            model.compile(loss='categorical_crossentropy',
                          optimizer='rmsprop',
                          metrics=['accuracy'])

        model.fit(X_train,
                  Y_train,
                  batch_size=batch_size,
                  validation_split=0.33,
                  epochs=epochs)
        model.save(args.model_filename)
    else:
        model = load_model(args.model_filename)

    # Evaluate
    print(confusion_matrix(Y_test, model.predict(X_test), tag_test))
