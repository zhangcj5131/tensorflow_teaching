from sklearn.datasets import load_files
from keras.utils import np_utils
import numpy as np
from glob import glob
import os
#查看是否安装了 torch:conda list torch
#如何安装 torch:https://pytorch.org/get-started/locally/
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
        self.train_dir = '/Users/cjz/data/dogImages/train'
        self.valid_dir = '/Users/cjz/data/dogImages/valid'
        self.test_dir = '/Users/cjz/data/dogImages/test'
        self.batch_size = 4

class Samples:
    def __init__(self, config: Config):
        self.config = config

    def load_data(self, path):
        data = load_files(path)
        dog_files = data['filenames']
        targets = data['target']
        target_names = data['target_names']
        # print(dog_files)
        # print(targets)
        # print(target_names)
        return dog_files, targets

    def data_explor(self):
        train_files, train_targets = self.load_data(self.config.train_dir)
        valid_files, valid_targets = self.load_data(self.config.valid_dir)
        test_files, test_targets = self.load_data(self.config.test_dir)

        dog_names = [item[item.index('.')+1:-1] for item in sorted(glob('/Users/cjz/data/dogImages/train/*/'))]
        print(dog_names)
        print('number of dog types:', len(dog_names))
        print('total images:', len(np.hstack([train_files, valid_files, test_files])))
        print('trian size:', len(train_files), 'valid size:', len(valid_files), 'test_size:', len(test_files))

    def data_transform(self, batch_size = 4):
        '''
        transforms.Resize(254):把数据大小改为 254*254*3
        transforms.RandomCrop(224):随机裁剪 224*224*3
        transforms.RandomHorizontalFlip():随机水平翻转
        transforms.RandomRotation(20):随机旋转 20 度
        transforms.ToTensor():把数据转成 tensor,会自动把数据大小修改到[0,1]
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]):在图片所有维度上,把数据改为 mean=0.5,std=0.5
        transforms.CenterCrop(224):从图片中间剪切 224*224*3


        '''
        train_transform = transforms.Compose(
            [transforms.Resize(254), transforms.RandomCrop(224),
             transforms.RandomHorizontalFlip(), transforms.RandomRotation(20),
             transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
        )
        # 注意,只有测试集做数据增强,验证集和测试集不做数据增强,这里裁剪是中间裁剪
        valid_transform = transforms.Compose(
            [transforms.Resize(254), transforms.CenterCrop(224), transforms.ToTensor(),
             transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
        )
        test_transform = transforms.Compose(
            [transforms.Resize(254), transforms.CenterCrop(224), transforms.ToTensor(),
             transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
        )

        train_data = datasets.ImageFolder(self.config.train_dir, transform=train_transform)
        valid_data = datasets.ImageFolder(self.config.valid_dir, transform=valid_transform)
        test_data = datasets.ImageFolder(self.config.test_dir, transform=test_transform)

        '''
        shuffle=True:打乱数据顺序
        构造迭代器,使得每次可以去到 batch_size 个数据
        '''
        train_dataloader = torch.utils.data.DataLoader(
            train_data, batch_size=batch_size, shuffle=True
        )
        valid_dataloader = torch.utils.data.DataLoader(
            valid_data, batch_size=batch_size, shuffle=True
        )
        test_dataloader = torch.utils.data.DataLoader(
            test_data, batch_size=batch_size, shuffle=True
        )
        return train_dataloader, valid_dataloader, test_dataloader

    def show_data(self):
        train_dataloader, valid_dataloader, test_dataloader = self.data_transform(batch_size=2)
        for index, (data, target) in enumerate(train_dataloader):
            if index > 0:
                break
            #默认拿到的数据都是 torch 里的 tensor,需要转成 numpy
            batch_images = data.numpy()#torch 的疏忽类型是[N, C, H, W]
            batch_images = np.transpose(batch_images, (0, 2, 3, 1))
            target = target.numpy()
            print(batch_images.shape)
            print('train type:', type(batch_images), 'train shape:', batch_images.shape,
                  'train min:', batch_images.min(), 'train max:', batch_images.max())
            print('labels:', target)

    def show_dog_images(self):
        train_dataloader, valid_dataloader, test_dataloader = self.data_transform(batch_size=32)

        def imshow(img):
            img = img / 2 + 0.5#troch 里数据的格式是:[C, H, W],需要装成[H, W, C]才能显示
            plt.imshow(np.transpose(img, [1,2,0]))

        dataiter = iter(train_dataloader)
        images, labels = dataiter.next()
        print(images.shape)
        images = images.numpy()
        #labels = labels.numpy()

        fig = plt.figure(figsize=(25, 4))
        for id in range(20):
            ax = fig.add_subplot(2, 10, id + 1, xticks=[], yticks=[])
            imshow(images[id])
            ax.set_title(labels[id].item())
        plt.show()





if __name__ == '__main__':
    config = Config()
    s = Samples(config)
    s.show_dog_images()