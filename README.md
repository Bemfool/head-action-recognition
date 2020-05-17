# Head Action Recognition
Use facial landmarks to classify head action (using LSTM)

### Environment

* Python 3.6
* Tensorflow 1.12.0
* Keras 2.2.4
* Pandas 0.20.3

### Network

* Masking		Process variable sequence length;
* LSTM
  * Hidden unit: 256
  * Max input sequence size: 60 frames
  * Input dimension: N_LANDMARK(68) * 2
* Dropout
* Output:     6 classes

### Usage

`har-angles.py [-h][-t] [-m MODEL_FILENAME][-d DATA_DIR]`

Optional arguments:

* -h, --help            show this help m essage and exit
* -t, --is_training     Set training mode
* -m MODEL_FILENAME, --model_filename MODEL_FILENAME
  ​                         Filename of the model to be loaded
* -d DATA_DIR, --data_dir DATA_DIR
  ​                        Directory of data  