import tensorflow as tf
import numpy as np
import matplotlib as plt
import pandas as pd

from tensorflow.keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data();

print("Training data shape: ", x_train.shape);
print("Testing data shape: ", x_test.shape);