# -- encoding:utf-8 --


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


# 打印numpy的数组对象的时候，中间不省略
np.set_printoptions(threshold=np.inf)


def show_image(image):
    shape = np.shape(image)
    if len(shape) == 3 and shape[2] == 1:
        # 黑白图像
        plt.imshow(image[:, :, 0], cmap='gray')
        plt.show()
    elif len(shape) == 3:
        # 彩色图像
        plt.imshow(image)
        plt.show()


# 1. 启动一个交互式的会话
sess = tf.InteractiveSession()

# todo 2. 读取图像数据
image_path = "./images/1.png"
#注意,这是一个 gif 动态图,有专门的方法处理它
# image_path = "./images/timg.gif"

"""
def read_file(filename, name=None):
  filename：给定一个待读取文件的路径
"""

file_contents = tf.read_file(image_path)
#打印图片的具体内容,都是一些 16 进制数据
print(file_contents.eval())#sess.run(file_contents)
print('--' * 40)

# todo 将数据转换为图像数据
"""
def decode_image(contents, channels=None, name=None):
    将图像数据转换为像素点的数据格式，返回对象为: [height, width, num_channels], 
                 如果是gif的图像返回[num_frames, height, width, num_channels]
        height: 图片的高度的像素大小
        width: 图片的水平宽度的像素大小
        num_channels: 图像的通道数，也就是API中的channels的值
        num_frames: 因为gif的图像是一个动态图像，可以将每一个动的画面看成一个静态图像，num_frames相当于在这个gif图像中有多少个静态图像
    一、contents: 给定具体的数据对象
    二、参数channels：可选值：0 1 3 4，默认为0， 一般使用0 1 3，不建议使用4
        0：使用图像的默认通道，也就是图像是几通道的就使用几通道
        1：使用灰度级别的图像数据作为返回值（只有一个通道：黑白）
        3：使用RGB三通道读取数据
        4：使用RGBA四通道读取数据(R：红色，G：绿色，B：蓝色，A：透明度)
"""
#直接看一下图片的代码
# image_tensor = tf.image.decode_image(contents=file_contents, channels=3)
# show_image(image_tensor.eval())

#注意,这个是只解析 png 格式的图片
image_tensor = tf.image.decode_png(contents=file_contents, channels=3, dtype=tf.uint8)
print('原始数据shape is:{}'.format(image_tensor.eval().shape))
show_image(image_tensor.eval())


#这个方法专门处理 gif 格式的动态图
# image_tensor = tf.image.decode_gif(contents=file_contents)
# # print(image_tensor)
# # print(image_tensor.eval())
# print("原始数据形状:{}".format(np.shape(image_tensor.eval())))
# show_image(image_tensor.eval()[6])


# todo 3. 图像大小的缩放
"""
def resize_images(images,
                  size,
                  method=ResizeMethod.BILINEAR,
                  align_corners=False):
    重置大小，放大或者缩小
        images: 给定需要进行大小重置的tensor对象，shape要求为: [batch_size, height, width, channel] 或者 [height, width, channel]； 表示可以一次对很多图像做大小重置，也可以仅仅对一个图像做一个大小重置操作；
        size：给定一个二元组，也就是(new_height, new_width)
        method: 做一个放大和缩小的时候，采用什么算法放大缩小（如何产生新的像素点的值）
            class ResizeMethod(object):
              BILINEAR = 0 # 默认值，二次线性插值
              NEAREST_NEIGHBOR = 1 # 使用邻居的像素值作为新的像素值
              BICUBIC = 2 # 三次插值，一般建议使用BICUBIC，但是运行速度比较慢。效果最好
              AREA = 3 # 使用一个区域的所有颜色的均值作为新的像素值,是 cv2 采用的
    返回的数据类型和输入的images的数据shape格式一致
    下面三个方法演示了三种不同的缩放图片的算法.大概了解一下
"""
# resize_image_tensor = tf.image.resize_images(
#     images=image_tensor, size=(128, 80),
#     method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

# resize_image_tensor = tf.image.resize_images(
#     images=image_tensor, size=(128, 80),
#     method=tf.image.ResizeMethod.AREA)

# float_image_tensor = tf.image.convert_image_dtype(image_tensor, dtype=tf.float32)
# resize_image_tensor = tf.image.resize_images(
#     images=float_image_tensor, size=(128, 80), method=3)
#
# print("rrize后的数据形状:{}".format(resize_image_tensor.eval().shape))
# show_image(resize_image_tensor.eval())


# todo 4. 图像的剪切和填充
# 图像剪切+填充+大小重置，如果给定大小小于原始图像的大小，那么进行剪切操作，如果给定的大小大于原始图像的大小，那么进行填充操作
"""
def resize_image_with_crop_or_pad(image, target_height, target_width):
    image：需要进行操作的图像tensor对象
    target_height, target_width: 新图像的高度和宽度
做填充和剪切的时候，是从中心位置开始计算,不支持任意位置的剪切
"""
# crop_or_pad_image_tensor = tf.image.resize_image_with_crop_or_pad(image_tensor,
#                                                                   target_height=800,
#                                                                   target_width=200)
# print("crop后的数据形状:{}".format(np.shape(crop_or_pad_image_tensor.eval())))
# show_image(crop_or_pad_image_tensor.eval())


# 从中心位置等比例的剪切,只支持从中心点剪切,这里剪切 60%
# central_crop_image_tensor = tf.image.central_crop(image_tensor, central_fraction=0.6)
# print("central_crop后的数据形状:{}".format(np.shape(central_crop_image_tensor.eval())))
# show_image(central_crop_image_tensor.eval())


# 基于给定的坐标进行数据的剪切
"""
def crop_to_bounding_box(image, offset_height, offset_width, target_height,
                         target_width):
        offset_height：给定从高度这个位置进行剪切，其实给定的是剪切的左上角的像素下标
        offset_width: 给定从宽度这个维度进行剪切，其实给定的是剪切的左上角的像素下标
"""
# crop_to_bounding_box_image_tensor = tf.image.crop_to_bounding_box(
#     image_tensor, 100, 20, 500, 490
# )
# print("crop_to_bounding后数据形状:{}".format(np.shape(crop_to_bounding_box_image_tensor.eval())))
# show_image(crop_to_bounding_box_image_tensor.eval())





# todo 5. 旋转
# 上下交换
# flip_up_down_image_tensor = tf.image.flip_up_down(image_tensor)
# print("flip_up_down后形状:{}".format(np.shape(flip_up_down_image_tensor.eval())))
# show_image(flip_up_down_image_tensor.eval())


# 左右交换
# flip_left_right_image_tensor = tf.image.flip_left_right(image_tensor)
# print("新的数据形状:{}".format(np.shape(flip_left_right_image_tensor.eval())))
# show_image(flip_left_right_image_tensor.eval())


# 转置(行，列互换)
# transpose_image_tensor = tf.image.transpose_image(image_tensor)
# print("transpose_image后形状:{}".format(np.shape(transpose_image_tensor.eval())))
# show_image(transpose_image_tensor.eval())


# 旋转（90、180、270、360）,只支持这四个度的旋转,k=1 就是 90 度
# random_int = np.random.randint(low=0, high=3)
# rot90_image_tensor = tf.image.rot90(image_tensor, k=random_int)
# print("新的数据形状:{}".format(np.shape(rot90_image_tensor.eval())))
# show_image(rot90_image_tensor.eval())


# todo 6. 颜色空间的转换
# NOTE: 如果要进行颜色空间的转换，那么必须将Tensor对象中的数据类型转换为float类型
# NOTE: 对于图像像素点的表示来讲，可以使用0~255的uint8类型的数值表示，也可以使用0~1之间的float类型的数据表示
# print(image_tensor.eval())
#做颜色转换的花,一定需要把tensor 转成 float 类型,默认是 int8
float_image_tensor = tf.image.convert_image_dtype(image_tensor, dtype=tf.float32)
# print(float_image_tensor.eval())

# RGB -> Gray,彩色转灰度图
gray_image_tensor = tf.image.rgb_to_grayscale(float_image_tensor)
# print("新的数据形状:{}".format(np.shape(gray_image_tensor.eval())))
# show_image(gray_image_tensor.eval())


# RGB -> HSV(RGB: 颜色是由三原色构成的，也就是R红色、G绿色、B蓝色；HSV：描述的是颜色的色彩信息，H：图像的色彩、色度，S表示的图像的饱和度；V表示亮度)
# 注意：必须使用float数据类型的图片，若用uint8的会报如下错误：
#    TypeError: Value passed to parameter 'images' has DataType uint8 not in list of allowed values: float32, float64
# hsv_image_tensor = tf.image.rgb_to_hsv(float_image_tensor)
# print("新的数据形状:{}".format(np.shape(hsv_image_tensor.eval())))
# # hsv的图像展示不是特别好...
# show_image(hsv_image_tensor.eval())

# hsv -> rgb
# rgb_image_tensor = tf.image.hsv_to_rgb(hsv_image_tensor)
# print("新的数据形状:{}".format(np.shape(rgb_image_tensor.eval())))
# show_image(rgb_image_tensor.eval())


# todo gray -> rgb 注意：只是通道增加了，并不能转为彩色
# rgb_image_tensor = tf.image.grayscale_to_rgb(gray_image_tensor)
# print("新的数据形状:{}".format(np.shape(rgb_image_tensor.eval())))
# show_image(rgb_image_tensor.eval())


# todo 灰度图作用：可以从颜色空间中提取图像的轮廓信息(图像的二值化)
# 图像的二值化
a = gray_image_tensor
b = tf.less_equal(a, 0.9)

# 0就是黑，1就是白
"""
def where(condition, x=None, y=None, name=None):
      condition: 给定一个bool数据组成的tensor对象
      x：当condition中的值为true的时候，返回的值
      y：当condition中的值为false的时候，返回的值
      NOTE: 要求condition、x、y的数据形状是一致的
"""
# 对于a中所有大于0.9的像素，设置为0，小于等于0.9的像素值设置为原始值
#tf.cond
#c = tf.where(condition=b, x=a, y=a - a)   # todo 注意y 这里只能用 a-a ，而不能直接用0，因为是一个矩阵
# show_image(c.eval())


# 对于a中所有小于等于0.9的像素，设置为1，大于0.9的像素设置为c的值
# d = tf.where(condition=b, x=tf.ones_like(c), y=c)
# print("新的数据形状:{}".format(np.shape(d.eval())))
# show_image(d.eval())


# todo 7. 图像的调整
# 亮度调整
"""
def adjust_brightness(image, delta):
  image: 需要调整的图像tensor对象
  delta：调整的参数值，取值范围:(-1,1); 该值表示亮度增加或者减少的值。
底层是将image转换为hsv格式的数据，然后再进行处理。# rgb -> hsv -> h,s,v+delta -> rgb
"""
# adjust_brightness_image_tensor = tf.image.adjust_brightness(image_tensor, delta=-0.5)
# print("新的数据形状:{}".format(np.shape(adjust_brightness_image_tensor.eval())))
# show_image(adjust_brightness_image_tensor.eval())


# 色调调整
# delta： 调整的参数值，取值范围:(-1,1); 该值表示色调增加或者减少的值。
# adjust_hue_image_tensor = tf.image.adjust_hue(image_tensor, delta=-0.8)
# # rgb -> hsv -> h+delta,s,v -> rgb
# print("新的数据形状:{}".format(np.shape(adjust_hue_image_tensor.eval())))
# show_image(adjust_hue_image_tensor.eval())


# 饱和度调整
# saturation_factor： 饱和度系数值
# rgb -> hsv -> h,s*saturation_factor,v -> rgb
# adjust_saturation_image_tensor = tf.image.adjust_saturation(image_tensor, saturation_factor=20)
# print("新的数据形状:{}".format(np.shape(adjust_saturation_image_tensor.eval())))
# show_image(adjust_saturation_image_tensor.eval())


# 对比度调整(在每一个通道上，让通道上的像素值调整)
# 底层计算：(x-mean) * contrast_factor + mean
# adjust_contrast_image_tensor = tf.image.adjust_contrast(image_tensor, contrast_factor=100)
# print("新的数据形状:{}".format(np.shape(adjust_contrast_image_tensor.eval())))
# show_image(adjust_contrast_image_tensor.eval())


# 图像的校验(要求输出的图像必须是浮点型的)
# 用途：检出图像信号中的深色部分和浅色部分，并使两者比例增大，从而提高图像的对比度
# adjust_gamma_image_tensor = tf.image.adjust_gamma(float_image_tensor, gamma=100)
# print("新的数据形状:{}".format(np.shape(adjust_gamma_image_tensor.eval())))
# show_image(adjust_gamma_image_tensor.eval())


# 图像的归一化API（只能每次对一张图像做归一化操作）
# per_image_standardization_image_tensor = tf.image.per_image_standardization(image_tensor)
# print("新的数据形状:{}".format(np.shape(per_image_standardization_image_tensor.eval())))
# show_image(per_image_standardization_image_tensor.eval())


# 给图像加一个噪音
noisy_image_tensor = image_tensor + tf.cast(5 * tf.random_normal(shape=[1944, 2054, 3], mean=0, stddev=0.1), tf.uint8)
# print("新的数据形状:{}".format(np.shape(noisy_image_tensor.eval())))
# show_image(noisy_image_tensor.eval())


# 将上面转换步骤模型图，保存
# writer = tf.summary.FileWriter("./model/test03", sess.graph)
# writer.close()


# todo 图像保存(基于scipy的相关API做图像处理)
from scipy.misc import imsave, imread, imresize, imshow, imfilter, imrotate

file_path = "./images/1.png"
img = imread(file_path)
img = imresize(img, size=(200, 200))
imsave('./images/11111.png', img)
print(type(img))
print(np.shape(img))
imsave('./images/test.png', noisy_image_tensor.eval())
print('done!!')
