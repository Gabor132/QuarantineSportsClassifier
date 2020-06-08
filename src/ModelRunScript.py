import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model
import os

os.environ['TF_KERAS'] = '1'

import keras2onnx
from os import listdir
from os.path import isfile, join

IN_COLAB = False
try:
    import google.colab

    IN_COLAB = True
    print("Running on Google Colab")
except:
    IN_COLAB = False
    print("Running on local machine")

print("Using tensorflow version {}".format(tf.__version__))
print("Using keras2onnx version {}. Make sure it is 1.7.0".format(keras2onnx.__version__))
learning_rate = 0.01
testing_data_percentage = 0.2
nr_of_frames_per_sequence = 15
nr_of_keypoints = 25
nr_of_values_per_keypoint = 3

# Dataset paths
DRAGOS_COLAB_DATASET_PATH = "/content/drive/My Drive/QuarantineSportsDatasets/Dataset/OpenPoseDataset/"
LOCAL_DATASET_PATH = "../datasets/"

# Model paths
DRAGOS_COLAB_MODEL_PATH = "/content/drive/My Drive/QuarantineSportsDatasets/Model/"
LOCAL_MODEL_PATH = "../"

current_dataset_path = DRAGOS_COLAB_DATASET_PATH if IN_COLAB else LOCAL_DATASET_PATH
current_model_path = DRAGOS_COLAB_MODEL_PATH if IN_COLAB else LOCAL_MODEL_PATH

# Get all file names
all_files_names = [f for f in listdir(current_dataset_path) if
                   isfile(join(current_dataset_path, f)) and f.endswith('.json')]
print("Following dataset files have been found: {}".format(all_files_names))

dataset_paths = []
for path in all_files_names:
    dataset_path = current_dataset_path + path
    dataset_paths.append(dataset_path)

print("All files found: {}".format(dataset_paths))

y_total = None
x_total = None
for (index, path) in enumerate(dataset_paths):
    print("For file at {}".format(path))
    df = pd.read_json(path)
    keypoints = df['Keypoints'].values
    file_y = df['Category'].values
    file_x = []
    for k in keypoints:
        if k is not None:
            newK = np.reshape(np.asarray(k), (25, 3))
            file_x.append(newK)
        else:
            file_x.append(np.reshape(np.zeros(75), (25, 3)))
    file_x = np.array(file_x)
    print("For file at {} found {} frames".format(path, file_y.shape[0]))
    if np.all(x_total is None):
        x_total = file_x
    else:
        x_total = np.vstack((x_total, file_x))
    if np.all(y_total is None):
        y_total = file_y
    else:
        y_total = np.hstack((y_total, file_y))
print("Total Frames: {}".format(y_total.shape[0]))

categories = np.unique(y_total)
print("Found {} categories".format(categories))

data_by_category = {}
for category in categories:
    # Get Indexes
    y_category_indexes = np.where(y_total == category)
    # Get Values
    x_category = x_total[y_category_indexes]
    data_by_category.update({category: x_category})
    print("Category {} has {} elements".format(category, len(x_category)))

# Check if there are any frames with no keypoints
empty_frame = np.zeros((25, 3))
for category in data_by_category.keys():
    x = data_by_category[category]
    empty_frame_indexes = []
    for index, x_value in enumerate(x):
        if np.array_equal(x_value, empty_frame):
            empty_frame_indexes.append(index)
    print("Category {} has {} empty frames to delete".format(category, len(empty_frame_indexes)))
    x = np.delete(x, empty_frame_indexes, axis=0)
    data_by_category.update({category: x})


def split_dataset_categories(dataset_dict):
    x_train_total = None
    x_test_total = None
    x_validate_total = None
    y_train_total = None
    y_test_total = None
    y_validate_total = None
    for aux_category in dataset_dict.keys():
        x = dataset_dict[aux_category]
        y = to_categorical(np.ones((x.shape[0], 1)) * aux_category, 4)
        aux_x_train, aux_x_test, aux_y_train, aux_y_test = train_test_split(x, y, test_size=0.2)
        aux_x_train, aux_x_validate, aux_y_train, aux_y_validate = train_test_split(aux_x_train, aux_y_train,
                                                                                    test_size=0.2)
        if x_train_total is None:
            x_train_total = aux_x_train
            x_test_total = aux_x_test
            x_validate_total = aux_x_validate
            y_train_total = aux_y_train
            y_test_total = aux_y_test
            y_validate_total = aux_y_validate
        else:
            x_train_total = np.concatenate((x_train_total, aux_x_train), axis=0)
            x_test_total = np.concatenate((x_test_total, aux_x_test), axis=0)
            x_validate_total = np.concatenate((x_validate_total, aux_x_validate), axis=0)
            y_train_total = np.concatenate((y_train_total, aux_y_train), axis=0)
            y_test_total = np.concatenate((y_test_total, aux_y_test), axis=0)
            y_validate_total = np.concatenate((y_validate_total, aux_y_validate), axis=0)
    return x_train_total, x_test_total, x_validate_total, y_train_total, y_test_total, y_validate_total


x_train, x_test, x_validate, y_train, y_test, y_validate = split_dataset_categories(data_by_category)
print("Number of training frame sets {} and number of testing frame sets {} and validation frame sets {}"
      .format(len(x_train), len(x_test), len(x_validate)))

model = tf.keras.Sequential()
model.add(tf.keras.Input(shape=(25, 3)))
model.add(tf.keras.layers.LSTM(units=75))
model.add(tf.keras.layers.ReLU())
model.add(tf.keras.layers.Dense(categories.shape[0]))
model.add(tf.keras.layers.Softmax(1))
model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=learning_rate), loss='categorical_crossentropy',
              metrics=[tf.keras.metrics.categorical_crossentropy], run_eagerly=True)
model.summary()
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

# Training
print("\n# Training")
model.fit(x_train, y_train, epochs=100, batch_size=50, verbose=1, validation_data=(x_validate, y_validate))

print('\n# Evaluate')
result = model.evaluate(x_test, y_test, batch_size=1, verbose=1)
print(result)
dict(zip(model.metrics_names, result))

# Predictions
predict = model.predict(x_test)
nr_correct = {0: 0, 1: 0, 2: 0, 3: 0}
nr_wrong = {0: 0, 1: 0, 2: 0, 3: 0}
for (index, p) in enumerate(predict):
    p_category = np.where(p == np.max(p))[0][0]
    e_category = np.where(y_test[index] == np.max(y_test[index]))[0][0]
    if p_category == e_category:
        nr_correct.update({e_category: nr_correct[e_category] + 1})
    else:
        nr_wrong.update({e_category: nr_wrong[e_category] + 1})
print("Total tested {}".format(len(predict)))
for c in nr_correct.keys():
    print("{} Correct/Wrong {}/{} - {}%".format(c, nr_correct[c],
                                                nr_wrong[c] + nr_correct[c],
                                                int(100 * nr_correct[c] / (nr_correct[c] + nr_wrong[c]))))

# serialize model to JSON
model_json = model.to_json()
with open(current_model_path + "model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights(current_model_path + "model.h5")
print("Saved model to disk")

onnx_model = keras2onnx.convert_keras(model)
keras2onnx.save_model(onnx_model, current_model_path + "model.onnx")
