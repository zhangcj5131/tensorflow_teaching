from sklearn.datasets import load_files
from keras.utils import np_utils
import numpy as np
from glob import glob
import os
from torchvision import datasets
import torch
from PIL import Image
import matplotlib as mpl
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class Config:
    def __init__(self):
        self.train_dir = '/Users/cjz/deep_learning/deep_learning_basic/03CNN/dogImages/train'
        self.valid_dir = '/Users/cjz/deep_learning/deep_learning_basic/03CNN/dogImages/valid'
        self.test_dir = '/Users/cjz/deep_learning/deep_learning_basic/03CNN/dogImages/test'
        self.batch_size = 4


class Samples:
    def __init__(self, config: Config):
        self.config = config

    def load_dataset(self, path):
        data = load_files(path)
        dog_files = data['filenames']
        targets = data['target']
        return dog_files, targets

    def explor_data(self):
        config = self.config
        train_files, tran_targets = self.load_dataset(config.train_dir)













if __name__ == '__main__':
    config = Config()
    s = Samples(config)
    s.load_dataset(config.train_dir)