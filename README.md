# Head Action Recognition
Use reprocessed data (rotation information) to classify head action (using LSTM)

### Environment

* Python 3.6
* Tensorflow 1.12.0
* Keras 2.2.4
* Pandas 0.20.3

### Data

Process video/camera by using [program(hpe_webcam)](<https://github.com/Great-Keith/head-pose-estimation/blob/master/cpp/hpe_webcam.cpp>) in <https://github.com/Great-Keith/head-pose-estimation> and save text file in current `./data` folder.

format: `yaw roll pitch tx ty tz`

*[Note] But tx, ty and tz would not be used.*

### Network

* Masking		Process variable sequence length;
* LSTM
  * Hidden unit: 256
  * Max input sequence size: 30 frames
  * Input dimension: 3 (yaw roll pitch)
* Dropout
* Output:     3 classes
  * still		Keep still
  * nod         Nod
  * shake      Shake head

### Usage

`har.py [-h][-t] [-m MODEL_FILENAME][-d DATA_DIR] [-s]`

Optional arguments:

* -h, --help            show this help m essage and exit
* -t, --is_training     Set training mode
* -m MODEL_FILENAME, --model_filename MODEL_FILENAME
  ​                         Filename of the model to be loaded
* -d DATA_DIR, --data_dir DATA_DIR
  ​                        Directory of data
* -s, --gen_syn_data    Except for data loaded, generate some new data  