import pandas as pd
import os
import xml.etree.cElementTree as et
import numpy as np

JPEGImages_path = r'G:\program\butterflyDetection\Faster-RCNN-tensorflow-master\data\VOCdevkit2007\VOC2007_base\JPEGImages/'
Annotation_path = r'G:\program\butterflyDetection\Faster-RCNN-tensorflow-master\data\VOCdevkit2007\VOC2007\Annotations/'
index = pd.read_csv('lable_index.txt', header=None)
index = index[0].tolist()

def get_alllabels(file_path):
    #处理xml得到所有图片的标签以及盒子
    all_labels = {}
    # all_boxs = {}
    i = 0
    for file in os.listdir(file_path):
        all_labels[i] = []
        # all_boxs[i]=[]
        tree = et.parse(file_path + file)
        root = tree.getroot()
        Object = root.findall('object')
        for obj in Object:
            name = obj.find('name').text
            all_labels[i].append(name)
            # 存盒子
            # bndbox = obj.find('bndbox')
            # xmin = bndbox.find('xmin').text
            # ymin = bndbox.find('ymin').text
            # xmax = bndbox.find('xmax').text
            # ymax = bndbox.find('ymax').text
            # all_boxs[i].append([xmin, ymin, xmax, ymax])
        # print(i)
        # print(all_labels[i])
        i += 1
    return all_labels

get_alllabels(Annotation_path)

all_labels = get_alllabels(Annotation_path)
df = pd.DataFrame(index=index, columns=['num'])

for i in range(94):
    label = index[i]
    num = 0
    for j in all_labels:
        if label in all_labels[j]:
            num += 1
    df.loc[label, 'num'] = num
df.to_csv('num.csv')
