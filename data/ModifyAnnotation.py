import xml.etree.ElementTree as et
import os

#修改所有文件的名称
def get_label(file_dir):
    i = 1
    for file in os.listdir(file_dir):
        n = 6 - len(str(i))
        tree = et.parse(file_dir + '\\' + file)
        root = tree.getroot()
        filename = root.find('filename')
        filename.text = str(0)*n + str(i) + '.jpg'

        tree.write(file_dir + '\\' + file, encoding = 'utf-8')
        i = i + 1

get_label('G:\program\Faster-RCNN-TensorFlow-Python3-master\data\VOCDevkit2007\VOC2007\Annotations')
