import tensorflow as tf

import tensorflow_datasets as tfds 

train, test = tfds.load('mnist', shuffle_files=True)