import os.path as osp
import sys
sys.path.append(osp.dirname(osp.dirname(__file__)))

from tensorflow.python.framework.dtypes import uint8

from numpy.testing._private.utils import verbose
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from collections import defaultdict
import pandas as pd
from shutil import copy
from tqdm import tqdm
import cv2

gpu = tf.config.experimental.list_physical_devices('GPU')[0]
tf.config.experimental.set_memory_growth(gpu, True)

image_size = (400, 120)
batch_size = 64

def process_path(file_path):
    #   label = np.array([0,1])
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.py_function(img_preprocessing, [img], uint8)
    img.set_shape(tf.TensorShape([None, None, 3]))
    img = tf.image.resize(img, image_size)
    return img

def img_preprocessing(img):
    h, w,_ = img.shape
    img = np.array(img)
    h_step = h//3
    w_step = w//3
    # img = cv2.vconcat([img[:h_step, :w_step], img[:h_step, -w_step:]])
    return img


pdf_files_ds = tf.data.Dataset.list_files('E:\\강서구 pdf 이미지\\*\\*.jpg')
capture_files_ds = tf.data.Dataset.list_files('E:\\capture\\*.jpg')

pdf_files_list = list(map(lambda x: x.decode('utf-8'), pdf_files_ds.take(-1).as_numpy_iterator()))
capture_files_list = list(map(lambda x: x.decode('utf-8'), capture_files_ds.take(-1).as_numpy_iterator()))

pdf_ds = pdf_files_ds.map(process_path, num_parallel_calls=batch_size).batch(batch_size)
capture_ds = capture_files_ds.map(process_path,num_parallel_calls=batch_size).batch(batch_size)

pdf_ds = pdf_ds.prefetch(buffer_size=batch_size)
capture_ds = capture_ds.prefetch(buffer_size=batch_size)

data_augmentation = keras.Sequential(
    [
        layers.experimental.preprocessing.Rescaling(1./255),
    ]
)

from tensorflow.keras.applications import NASNetMobile
from tensorflow.keras.layers.experimental import preprocessing

def build_model(img_size):
    inputs = layers.Input(shape=img_size+(3,))
    x = data_augmentation(inputs)
    model = NASNetMobile(include_top=False, input_tensor=x, weights="imagenet")
    outputs = tf.nn.softmax(model.output, axis=3)
    # x = model.layers[300].output
    # outputs = layers.GlobalAveragePooling2D()(x)
    # x = tf.math.reduce_mean(model.output, axis=3)
    model = keras.Model(inputs, outputs)
    return model

model:keras.Model = build_model(image_size)

pdf_filename = osp.join('E:\\', 'pdf.dat')
capture_filename= osp.join('E:\\', 'capture.dat')
output_shape = model.output_shape[-1]

def predict():
    pdf_pred = np.memmap(pdf_filename, dtype='float32', mode='w+', shape=(len(pdf_files_list), output_shape))
    # pdf_pred = np.zeros((len(pdf_files_list), output_shape))
    # start = 0
    pdf_pred = model.predict(pdf_ds, verbose=1)
    # for batch in tqdm(pdf_ds):
    #     file_num = batch.shape[0]
    #     pred = model.predict_on_batch(batch)
    #     pdf_pred[start:start+file_num,:] = pred[:]
    #     start+=file_num
    #     # pdf_pred.flush()

    capture_pred = np.memmap(capture_filename, dtype='float32', mode='w+', shape=(len(capture_files_list), output_shape))
    # capture_pred = np.zeros((len(pdf_files_list), output_shape))
    capture_pred = model.predict(capture_ds, verbose=1)
    # start = 0
    # for batch in tqdm(capture_ds):
    #     file_num = batch.shape[0]
    #     pred = model.predict_on_batch(batch)
    #     capture_pred[start:start+file_num,:] = pred[:]
    #     start+=file_num
    #     # capture_pred.flush()

    # np.save('E:\\pdf-1.npy', pdf_pred)
    # np.save('E:\\capture-1.npy', capture_pred)
    return pdf_pred, capture_pred


def img_match(pdf_pred, capture_pred, pdf_files_list, capture_files_list):
    # step = 50000
    # def loss_model():
    #     inputs = (layers.Input(shape=(len(pdf_pred[0]),)), layers.Input(shape=(len(pdf_pred[0]),)))
    #     input1, input2 = inputs
    #     loss = tf.math.reduce_mean(tf.math.squared_difference(input1, input2), 1)
    #     model = keras.Model(inputs, loss)
    #     return model

    # def capture_data_generator():
    #     return iter(capture_pred)

    matching = defaultdict(str)
    # loss_func = loss_model()
    # capture_feature_dataset = tf.data.Dataset.from_tensor_slices(capture_pred).batch(step)
    for pdf_idx, pdf_feature in enumerate(tqdm(pdf_pred)):
        # pdf_feature_dataset = tf.data.Dataset.from_tensors(pdf_feature).repeat(len(capture_files_list)).batch(step)
        # feature_dataset = tf.data.Dataset.zip((pdf_feature_dataset, capture_feature_dataset))
        # feature_dataset.prefetch(step)
        # pdf_feature_dataset.prefetch(step)
        # capture_feature_dataset.prefetch(step)

        loss = np.zeros(len(capture_files_list))
        # for idx, batch in enumerate(tqdm(feature_dataset, total=np.ceil(len(capture_files_list)/step))):
            # loss[idx*step:idx*step+batch[0].shape[0]] = loss_func.predict_on_batch(batch)
        # min_idx = tf.argmin(loss)

        # for s in tqdm(range(0,len(capture_files_list), step)):
        loss = np.mean(np.square(pdf_feature-capture_pred),axis=1)

        min_idx = np.argmin(loss)
        matching[pdf_files_list[pdf_idx]] = capture_files_list[min_idx]
        # if matching[pdf_files_list[pdf_idx]]['loss'] > loss:
        #     matching[pdf_files_list[pdf_idx]]['loss'] = loss
        #     matching[pdf_files_list[pdf_idx]]['file'] = capture_files_list[capture_idx]
        # for capture_idx, capture_feature in enumerate(tqdm(capture_pred)):
        #     loss = np.sqrt(np.mean(np.square(pdf_feature-capture_feature)))
        #     if matching[pdf_files_list[pdf_idx]]['loss'] > loss:
        #         matching[pdf_files_list[pdf_idx]]['loss'] = loss
        #         matching[pdf_files_list[pdf_idx]]['file'] = capture_files_list[capture_idx]

    copy_path = 'E:\\copy'
    if not osp.isdir(copy_path):
        os.makedirs(copy_path)

    for val in matching.values():
        filename = osp.basename(val)
        copy(val, osp.join(copy_path, filename))

    df = pd.Series(matching)
    df.to_csv('E:\\matching.csv')

if __name__ == "__main__":
    pdf_pred, capture_pred = predict()
    # pdf_pred = np.array(np.memmap(pdf_filename, dtype='float32', mode='r+', shape=(len(pdf_files_list), output_shape)))
    # capture_pred = np.array(np.memmap(capture_filename, dtype='float32', mode='r+', shape=(len(capture_files_list), output_shape)))
    # pdf_pred = np.load('E:\\pdf-1.npy')
    # capture_pred = np.load('E:\\capture-1.npy')
    # np_to_tfrecords(pdf_pred, None, 'E:\\pdf', 1)
    # np_to_tfrecords(capture_pred, None, 'E:\\capture', 1)
    img_match(pdf_pred, capture_pred, pdf_files_list, capture_files_list)