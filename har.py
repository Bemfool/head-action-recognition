import tensorflow as tf
from keras import backend as K
from keras.models import load_model
import pandas as pd
import numpy as np
from keras.layers import Masking
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers.core import Dense, Dropout
import input_data
import argparse
np.random.seed(42)
tf.set_random_seed(42)

sess = tf.Session(graph=tf.get_default_graph())
K.set_session(sess)

epochs = 500
batch_size = 64
n_hidden = 256
n_classes = 3

ACTIONS = {
    0: 'STILL',
    1: 'NOD',
    2: 'SHAKE',
}


def confusion_matrix(Y_true, Y_pred):
    print(Y_true)
    print(Y_pred)
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
        '-s',
        '--gen_syn_data',
        action="store_true",
        help='Except for data loaded, generate some new data'
    )

    args = parser.parse_args()
    data_set = input_data.read_data_sets(args.data_dir)

    if args.gen_syn_data:
        data_set.gen_syn_data(10, 10, 10)

    X_train, Y_train, X_test, Y_test = data_set.train_test_split()
    if args.is_training:

        epochs = 500
        batch_size = 64
        n_hidden = 256

        timesteps = len(X_train[0])
        input_dim = len(X_train[0][0])

        model = Sequential()
        model.add(Masking(mask_value=[0, 0, 0], input_shape=(timesteps, input_dim)))
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
    print(confusion_matrix(Y_test, model.predict(X_test)))
