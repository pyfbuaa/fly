#!/usr/bin/env python

# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import csv

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from lib.config import config as cfg
from lib.utils.nms_wrapper import nms
from lib.utils.test import im_detect
from lib.nets.vgg16 import vgg16
from lib.utils.timer import Timer

# 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']
#检测图片的地址
file_dir = '.\data/demo'

CLASSES = ('__background__',
           '巴黎翠凤蝶', '柑橘凤蝶', '玉带凤蝶', '碧凤蝶', '红基美凤蝶', '蓝凤蝶', '金裳凤蝶', '青凤蝶', '朴喙蝶', '密纹飒弄蝶', '小黄斑弄蝶',
            '无斑珂弄蝶', '直纹稻弄蝶', '花弄蝶', '隐纹谷弄蝶', '绢斑蝶', '虎斑蝶', '亮灰蝶', '咖灰蝶', '大紫琉璃灰蝶', '婀灰蝶', '曲纹紫灰蝶',
            '波太玄灰蝶', '玄灰蝶', '红灰蝶', '线灰蝶', '维纳斯眼灰蝶', '艳灰蝶', '蓝灰蝶', '青海红珠灰蝶', '古北拟酒眼蝶', '阿芬眼蝶', '拟稻眉眼蝶',
            '牧女珍眼蝶', '白眼蝶', '菩萨酒眼蝶', '西门珍眼蝶', '边纹黛眼蝶', '云粉蝶', '侏粉蝶', '大卫粉蝶', '大翅绢粉蝶', '宽边黄粉蝶', '山豆粉蝶',
            '橙黄豆粉蝶', '突角小粉蝶', '箭纹云粉蝶', '箭纹绢粉蝶', '红襟粉蝶', '绢粉蝶', '菜粉蝶', '镉黄迁粉蝶', '黎明豆粉蝶', '依帕绢蝶', '四川绢蝶',
            '珍珠绢蝶', '蛇目褐蚬蝶', '中环蛱蝶', '云豹蛱蝶', '伊诺小豹蛱蝶', '小红蛱蝶', '扬眉线蛱蝶', '斐豹蛱蝶', '曲斑珠蛱蝶', '柱菲蛱蝶', '柳紫闪蛱蝶',
            '灿福蛱蝶', '玄珠带蛱蝶', '珍蛱蝶', '琉璃蛱蝶', '白钩蛱蝶', '秀蛱蝶', '绢蛱蝶', '绿豹蛱蝶', '网蛱蝶', '美眼蛱蝶', '翠蓝眼蛱蝶', '老豹蛱蝶',
            '荨麻蛱蝶', '虬眉带蛱蝶', '蟾福蛱蝶', '钩翅眼蛱蝶', '银斑豹蛱蝶', '银豹蛱蝶', '链环蛱蝶', '锦瑟蛱蝶', '黄环蛱蝶', '黄钩蛱蝶', '黑网蛱蝶',
            '尖翅翠蛱蝶', '素弄蝶', '翠袖锯眼蝶', '蓝点紫斑蝶', '雅弄蝶')

NETS = {'vgg16': ('vgg16_faster_rcnn_iter_3000.ckpt',), 'res101': ('res101_faster_rcnn_iter_110000.ckpt',)}
DATASETS = {'pascal_voc': ('voc_2007_trainval',), 'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',)}


def vis_detections(ax, image_name, class_name, dets, thresh):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]

    if len(inds) == 0:
        return

    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
        )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

        # 保存测试结果
        with open("./Results/result.csv", "a", newline='', encoding='utf-8') as file:
            writer = csv.writer(file, delimiter=',')
            writer.writerow([image_name.split('.')[0], CLASSES.index(class_name)-1, score, bbox[0], bbox[1], bbox[2], bbox[3]])
            print("数据写入result文件完成...")

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name, thresh), fontsize=14)



def demo(sess, net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""
    # Load the demo image
    im_file = os.path.join(cfg.FLAGS2["data_dir"], 'demo', image_name)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)
    timer.toc()
    print('Detection took {:.3f}s'.format(timer.total_time))

    # Visualize detections for each class
    CONF_THRESH = 0.1
    NMS_THRESH = 0.1

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1  # because we skipped background
        cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        vis_detections(ax, image_name, cls, dets, thresh=CONF_THRESH)

        plt.draw()


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101]',
                        choices=NETS.keys(), default='vgg16')
    parser.add_argument('--dataset', dest='dataset', help='Trained dataset [pascal_voc pascal_voc_0712]',
                        choices=DATASETS.keys(), default='pascal_voc_0712')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    # model path
    demonet = args.demo_net
    dataset = args.dataset
    tfmodel = r'.\default\voc_2007_trainval\default\40000/vgg16_faster_rcnn_iter_40000.ckpt'

    if not os.path.isfile(tfmodel + '.meta'):
        print(tfmodel)
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly?').format(tfmodel + '.meta'))

    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True

    # init session
    sess = tf.Session(config=tfconfig)
    # load network
    if demonet == 'vgg16':
        net = vgg16(batch_size=1)
    else:
        raise NotImplementedError

    n_classes = len(CLASSES)
    # create the structure of the net having a certain shape (which depends on the number of classes)
    net.create_architecture(sess, "TEST", n_classes,
                            tag='default', anchor_scales=[8, 16, 32])
    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)

    print('Loaded network {:s}'.format(tfmodel))

    # FF = ['IMG_000013.jpg']
    # for file in FF:
    for file in os.listdir(file_dir):
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('Demo for data/demo/{}'.format(file))
        demo(sess, net, file)

        #保存检测结果的图片
        # plt.savefig('G:\program\ButterflyDetection_frcnn\Results\save_demo_test/' + file, format='jpg', transparent=True, pad_inches=0, dpi=300, bbox_inches='tight')

    # plt.show()