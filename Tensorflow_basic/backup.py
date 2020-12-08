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
        self.train_dir = '/Users/cjz/data/dogImages/train'
        self.valid_dir = '/Users/cjz/data/dogImages/valid'
        self.test_dir = '/Users/cjz/data/dogImages/test'
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
        valid_files, valid_targets = self.load_dataset(config.valid_dir)
        test_files, test_targets = self.load_dataset(config.test_dir)

        # print(train_files, tran_targets)

        # 2、加载狗的品种
        # dogname_index_from = item.index('.') + 1
        dog_names = [item[item.index('.') + 1: -1] for item in sorted(glob('/Users/cjz/data/dogImages/train/*/'))]
        print(dog_names)
        print(train_files)
        print('总的狗狗的类别数量:{}'.format(len(dog_names)))
        print('总共有图片数量:{}'.format(len(np.hstack([train_files, valid_files, test_files]))))
        print('train dog images:{} - Valid dog images:{} - Test dog images:'
              '{}'.format(len(train_files), len(valid_files), len(test_files)))

    def data_transform(self, batch_size=4):
        """
        实现：1、缩放 ； 2、裁剪为224*224*3 ；3、随机翻转； 4、随机旋转0-20度；5、标准化
        :param batch_size:
        :return:
        transforms.Resize(254):图片转成 254,254,3
        transforms.RandomCrop(224):图片随机裁剪成 224,224,3,
        transforms.RandomHorizontalFlip():图片随机水平翻转
        transforms.RandomRotation(20):图片随机旋转 0-20 度

        transforms.ToTensor():图片转为 tensor,这个过程会自动把数据转为[0-1]之间,
        所以后面 mena 和 std 都是 0.5,这样可以把[0,1]之间的数据转换到[-1,1]之间

        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]:对图片进行标准化,三个维度分别是 RGB,三个维度均值都是 0.5,方差也都是 0.5
        """

        # Normalize([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]) 指 在所有三个维度,全部mean=0.5, std = 0.5
        train_transform = transforms.Compose(

            [transforms.Resize(254), transforms.RandomCrop(224), transforms.RandomHorizontalFlip(),
             transforms.RandomRotation(20), transforms.ToTensor(),
             transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
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

        # 使用 image datasets 和 transforms 来定义一个数据加载的迭代器。
        train_dataloaders = torch.utils.data.DataLoader(
            train_data, batch_size=batch_size, shuffle=True
        )
        valid_dataloaders = torch.utils.data.DataLoader(
            valid_data, batch_size=batch_size, shuffle=True
        )
        test_dataloaders = torch.utils.data.DataLoader(
            test_data, batch_size=batch_size, shuffle=True
        )
        return train_dataloaders, valid_dataloaders, test_dataloaders

    def show_data(self):
        train_dataloaders, valid_dataloaders, test_dataloaders = self.data_transform(batch_size=2)
        for batch_idx, (data, target) in enumerate(train_dataloaders):
            batch_images = data.numpy()  # [N, C, H, W]
            batch_images = np.transpose(batch_images, axes=[0, 2, 3, 1])  # [N, H, W, C]
            batch_y = target.numpy()
            print('train type:', type(batch_images), "train shape:",batch_images.shape,
                  'train min value:', batch_images.min(), 'train max value:', batch_images.max())
            print('label:', batch_y, 'label shape:', batch_y.shape)  # [ 0 43] (2,) <class 'numpy.int64'>
            print('**' * 80)
            if batch_idx >= 2:
                break

    def show_dog_images(self):
        train_dataloaders, valid_dataloaders, test_dataloaders = self.data_transform(batch_size=32)


        def imshow(img):
            # 迭代器中的数据都是[-1,1]之间,需要转回[0,1]
            img = img / 2 + 0.5
            #torch 中每个图片格式是[C,H,W],需要换回[H, C, W]
            plt.imshow(np.transpose(img, (1, 2, 0)))

        # 定义一个迭代器
        dataiter = iter(train_dataloaders)
        #从迭代器中获得一组图片,batch_size 个
        images, labels = dataiter.next()
        print(images.shape)#torch.Size([32, 3, 224, 224])
        images = images.numpy()

        # 画图
        fig = plt.figure(figsize=(25, 4))
        for idx in np.arange(20):
            #构建一个子图 size,2 行,20 / 2列,当前图片是第idx + 1个
            ax = fig.add_subplot(2, 20 / 2, idx + 1, xticks=[], yticks=[])
            imshow(images[idx])
            ax.set_title(labels[idx].item())
        plt.show()


if __name__ == '__main__':
    config = Config()
    s = Samples(config)
    s.show_dog_images()










if __name__ == '__main__':
    config = Config()
    s = Samples(config)
    s.load_dataset(config.train_dir)