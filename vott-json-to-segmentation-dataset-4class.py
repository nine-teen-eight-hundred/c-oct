#! /usr/bin/env python3
# -*- coding: utf-8 -*-

################################################################################
## 
## https://github.com/divamgupta/image-segmentation-keras
## に使う学習データを作成するスクリプト。
##
## Usage:
##     python3 vott-json-to-segmentation-dataset.py {vott-target-dir} {vott-source-dir} {resize|center}
## 
## vott-target-dir: JSON ファイルがある場所
## vott-source-dir: 画像ファイルがある場所
##
################################################################################

import numpy as np
import cv2
import json
from collections import OrderedDict
import pprint
import sys
import os
import urllib.parse
import random
import glob

# データセット出力先ディレクトリ
dataset_dir = './dataset'

# タグとラベルの対応
tags = {
    "Fibrocalcific plaque": 1, "Fibrocalcific palque": 1,
    "Fibrous cap atheroma": 2, "TCFA": 2,
    "Healed erosion/rupture": 3,
    "Intimal xanthoma": 4, "Pathological intimal thickening": 4,
}

# 各クラスに割り当てる BGR カラー値。
# https://github.com/divamgupta/image-segmentation-keras#preparing-the-data-for-training
# B チャンネルの値がクラスのラベルとして使われる。
# G,R チャンネルは意味を持たないので、便宜上、目視確認しやすくするために使う。
colors = {
    0: (0, 000, 000),
    1: (1, 000, 255),
    2: (2, 127, 255),
    3: (3, 255, 255),
    4: (4, 255, 000),
}

# 出力画像サイズ
# https://divamgupta.com/image-segmentation/2019/06/06/deep-learning-semantic-segmentation-keras.html
#   Choosing the input size
#   Apart from choosing the architecture of the model, choosing the model input size is 
#   also very important. If there are a large number of objects in the image, the input 
#   size shall be larger. In some cases, if the input size is large, the model should 
#   have more layers to compensate. The standard input size is somewhere from 200x200 
#   to 600x600. A model with a large input size consumes more GPU memory and also would 
#   take more time to train.

# size = 608 # 32 で割り切れる数にしておく（VGG16, ResNet-50 対応）
size = 576 # 192 で割り切れる数にしておく（PSPNet 対応）

# 出力先ディレクトリ
output_x_train = dataset_dir + '/train_images/'
output_y_train = dataset_dir + '/train_annotations/'
output_x_val   = dataset_dir + '/val_images/'
output_y_val   = dataset_dir + '/val_annotations/'
output_x_test  = dataset_dir + '/test_images/'
output_y_test  = dataset_dir + '/test_annotations/'
output_preview = dataset_dir + '/preview/'
os.makedirs(output_x_train, exist_ok=True)
os.makedirs(output_y_train, exist_ok=True)
os.makedirs(output_x_val, exist_ok=True)
os.makedirs(output_y_val, exist_ok=True)
os.makedirs(output_x_test, exist_ok=True)
os.makedirs(output_y_test, exist_ok=True)
os.makedirs(output_preview, exist_ok=True)

# トレーニング用とテスト用の比率
split_test  = 0.1
split_val   = 0.1
split_train = 0.8
random.seed(777)

def read_json(filename, source_dir, resize=True):
    fp = open(filename)
    _d = json.load(fp)

    _asset = _d['asset']
    id     = _asset['id']
    name   = _asset['name']
    path   = _asset['path']
    width  = _asset['size']['width']
    height = _asset['size']['height']

    # URLエンコードされてるのでデコード
    name = urllib.parse.unquote(name)

    filename_orig = source_dir + name
    if os.path.exists(filename_orig) == False:
        print('[Wargning] ' + id + ' : skipped missing file: ' + name)
        return

    if resize:
        if (width > height):
            scale = size / width
        else:
            scale = size / height
        shift_x = 0
        shift_y = 0
    else:
        scale = 1 
        shift_x = (size - width) / 2  #(608-500)/2 = 54
        shift_y = (size - height) / 2 #(608-578)/2 = 15

    shape = [size, size, 3] # BGR
    img = np.zeros(shape, dtype=np.uint8)
    __tag_indexes = []

    for _region in _d['regions']:
        _tags    = _region['tags']
        _points  = _region['points']

        if len(_tags) == 0:
            sys.exit('len(tags) is 0')
        elif len(_tags) > 1:
            print('[Wargning] ' + id + ' : skipped multi-selected tags')

        _tag = _tags[0].strip()
        if _tag not in tags:
            print('[Info] ' + id + ' : skipped tag : ' + _tag)
            continue
        tag_index = tags[_tag]
        tag_color = colors[tag_index]

        __tag_indexes.append(tag_index)

        points = [[p['x'] * scale + shift_x, p['y'] * scale + shift_y] for p in _points]
        points = np.array(points).astype(np.int32)
        img = cv2.fillPoly(img, pts=[points], color=tag_color)
    
    _split_as = ''
    if random.random() < split_train / (split_train + split_val + split_test):
        output_y = output_y_train
        output_x = output_x_train
        _split_as = 'train'
    else:
        if random.random() < split_test / (split_test + split_val):
            output_y = output_y_test
            output_x = output_x_test
            _split_as = 'test'
        else:
            output_y = output_y_val
            output_x = output_x_val
            _split_as = 'val'


    # アノテーション画像
    filename_annotation = output_y + id + '.png'
    cv2.imwrite(filename_annotation, img)

    # RGB 画像（サイズ統一）
    img_orig = cv2.imread(filename_orig)
    if resize:
        img_orig = cv2.resize(img_orig, None, fx=scale, fy=scale)
        _h, _w = img_orig.shape[:2]
        img_dest = np.zeros(shape, dtype=np.uint8)
        img_dest[0:_h, 0:_w] = img_orig
    else:
        _border = 1
        img_orig = cv2.copyMakeBorder(img_orig, _border, _border, _border, _border, cv2.BORDER_CONSTANT, value=[0,0,0])
        _patch_size = (size, size)
        _center = (width/2 + _border, height/2 + _border)
        img_dest = cv2.getRectSubPix(img_orig, _patch_size, _center)

    filename_input = output_x + id + '.png' # JPEG 圧縮で劣化するのは嫌なので
    cv2.imwrite(filename_input, img_dest)

    # プレビュー画像
    img_preview = cv2.addWeighted(img_dest, 1.0, img, 0.25, 0)
    filename_preview = output_preview + id + '.jpg'
    cv2.imwrite(filename_preview, img_preview)

    # ファイル名の対応を出力
    print("{} : {} : {} : {}".format(id, filename_orig, _split_as, __tag_indexes))

# 実行用
if __name__ == '__main__':
    args = sys.argv
    if (len(args) != 4):
        exit('Usage: python3 vott-json-to-segmentation-dataset.py {vott-target-dir} {vott-source-dir} {resize|center}')

    vott_target_dir = args[1]
    vott_source_dir = args[2]
    mode = args[3]

    if (not(os.path.exists(vott_source_dir)) or not(os.path.isdir(vott_source_dir))):
        exit('source-dir does not exists')

    if (not(os.path.exists(vott_target_dir)) or not(os.path.isdir(vott_target_dir))):
        exit('target-dir does not exists')

    for json_file in glob.glob(vott_target_dir + "/*.json"):
        read_json(json_file, vott_source_dir, mode=='resize')
