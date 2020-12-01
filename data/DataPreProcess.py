import os
import csv
import json
import imghdr
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from numpy import *
import xml.etree.cElementTree as et


# 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']

train_list_path = r'G:\program\butterflyDetection\Faster-RCNN-tensorflow-master\data\VOCdevkit2007\VOC2007_base\JPEGImages'  # 初始训练样本数据的路径
label_list_path = r'G:\program\butterflyDetection\Faster-RCNN-tensorflow-master\data\VOCdevkit2007\VOC2007_base\Annotations' #标签的读入路径
test_list_path = 'G:\研究生\研一上\数据挖掘导论\Competitions\Data\TestData'  # 测试样本的读入路径


#处理蝴蝶名称的索引问题
def get_labelIndex():
    labelIndex = {}
    EnNameIndex = pd.read_table("G:\研究生\研一上\数据挖掘导论\Competitions\Data\\name_index\en_name_index.txt",
                                names=['EnName', 'Index'], header=None)
    ChName2EnName = pd.read_table("G:\研究生\研一上\数据挖掘导论\Competitions\Data\\name_index\cn_2_en_name.txt",
                                  names=['ChName', 'EnName'], header=None)
    NameIndex = pd.merge(EnNameIndex, ChName2EnName, how='inner')
    for i in range(NameIndex.shape[0]):
        labelIndex[NameIndex.loc[i, "ChName"]] = NameIndex.loc[i, "Index"]
    return labelIndex

#获取所有训练集图片的路径
def get_image_paths(file_dir):
    all_image_paths = []
    for file in os.listdir(file_dir):
        all_image_paths.append(file_dir + "\\" + file)
    # for i, file in enumerate(os.listdir(file_dir)):
        # with open(file_dir+'\\'+file, 'rb') as imageFile:
        #     if imageFile.read().startswith(b'II*'):
        #         print(f"{i}: {file} - found!")
    return all_image_paths

#加载和预处理图片
def load_and_preprocess_image(path):
    # 原始数据
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [32, 32])
    image = tf.cast(image, tf.float32)
    image /= 255.0  # normalize to [0,1] range
    return image

# #写中文标签的索引文档
def write_cn_index():
    label_to_index = get_labelIndex()
    with open(r'G:\program\ButterflyDetection_frcnn\cn_name_index.csv', 'a', encoding="utf-8") as f:
        writer = csv.writer(f)
        for item in label_to_index.items():
            writer.writerow(item)


