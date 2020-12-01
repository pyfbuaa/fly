# -*- coding:utf8 -*-

import os

#批量重命名文件夹中的图片文件
class BatchRename():

    def __init__(self):
        # 图片文件夹路径horse
        self.path = 'G:\program\Faster-RCNN-TensorFlow-Python3-master\data\VOCDevkit2007\VOC2007\Annotations'

    def rename(self):
        filelist = os.listdir(self.path)
        total_num = len(filelist)
        i = 1
        for item in filelist:
            if item.endswith('.xml'):
                n = 6 - len(str(i))
                src = os.path.join(os.path.abspath(self.path), item)
                dst = os.path.join(os.path.abspath(self.path), str(0)*n + str(i) + '.xml')
                try:
                    os.rename(src, dst)
                    print('converting %s to %s ...' % (src, dst))
                    i = i + 1

                except:
                    continue
        print('total %d to rename & converted %d jpgs' % (total_num, i))


if __name__ == "__main__":
    BatchRename = BatchRename()
    BatchRename.rename()