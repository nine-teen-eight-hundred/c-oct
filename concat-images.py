#! /usr/bin/env python3
# -*- coding: utf-8 -*-

########################################################################
## 
## Usage: concat-image.py {images-dir}
##
## 指定ディレクトリにある以下の画像を水平連結した画像を生成する。
## 
## - {name}.jpg                 テスト画像
## - {name}.png                 正解ラベル画像 (Ground Truth)
## - {name}.png.{model}.jpg     各モデルの出力画像
## 
## 必要な OpenCV と Numpy をインストールするには：
##    pip3 install opencv-python -t .
## 
## テスト画像が png なら `mogrify -format jpg -path {images-dir} *.png` などで変換すること。
##
########################################################################

import numpy as np
import cv2
import sys
import os
import glob

# models = []
# for i in [0,1,2,3,4,5,10,25,50]:
#     models.append({
#         'name': "resnet50_pspnet_ep{}".format(i),
#         'text': "Resnet50 x PSPNet ({})".format(i),
#     })

models = [
    # {
    #     'name': 'pspnet',
    #     'text': 'Vanila CNN x PSPNet',
    # },
    # {
    #     'name': 'vgg_pspnet',
    #     'text': 'VGG16 x PSPNet',
    # },
    {
        'name': 'resnet50_pspnet',
        'text': 'Resnet50 x PSPNet',
    },
    # {
    #     'name': 'segnet',
    #     'text': 'Vanila CNN x SegNet',
    # },
    # {
    #     'name': 'vgg_segnet',
    #     'text': 'VGG16 x SegNet',
    # },
    # {
    #     'name': 'resnet50_segnet',
    #     'text': 'Resnet50 x SegNet',
    # },
    # {
    #     'name': 'vgg_unet',
    #     'text': 'VGG16 x U-Net',
    # },
    # {
    #     'name': 'resnet50_unet',
    #     'text': 'Resnet50 x U-Net',
    # },
    # {
    #     'name': 'fcn_32',
    #     'text': 'Vanila CNN x FCN',
    # },
    # {
    #     'name': 'fcn_32_vgg',
    #     'text': 'VGG16 x FCN',
    # },
]

if __name__ == "__main__":
    args = sys.argv
    if len(args) != 2 :
        exit('Usage: concat-image.py {images-dir}')
    
    images_dir = args[1]

    names = []
    for afile in glob.glob(images_dir + '/*.jpg') :
        if '.png' in afile:
            continue
        name = os.path.splitext(os.path.basename(afile))[0]
        names.append(name)
    
    for name in names :

        ideal_image_path = images_dir + '/' + name + '.jpg'
        ideal_image = cv2.imread(ideal_image_path)

        cv2.putText(ideal_image, 'Source', (10, ideal_image.shape[0] - 5), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255))

        annotation_image_path = images_dir + '/' + name + '.png'
        annotation_image = cv2.imread(annotation_image_path)

        cv2.putText(annotation_image, 'Annotation', (10, annotation_image.shape[0] - 5), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255))

        images = [
            cv2.copyMakeBorder(ideal_image, 1, 1, 1, 1, cv2.BORDER_CONSTANT, None, [255,255,255]),        
            cv2.copyMakeBorder(annotation_image, 1, 1, 1, 1, cv2.BORDER_CONSTANT, None, [255,255,255]),        
        ]

        for index, model in enumerate(models) :
            model_name = model['name']
            model_text = model['text']

            image_path = images_dir + '/' + name + '.png.' + model_name + '.jpg'

            if os.path.exists(image_path) :
                model_image = cv2.imread(image_path)
            else:
                model_image = np.zeros(ideal_image.shape, dtype=np.uint8)
            
            cv2.putText(model_image, model_text, (10, model_image.shape[0] - 5), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255))

            model_image = cv2.copyMakeBorder(model_image, 1, 1, 1, 1, cv2.BORDER_CONSTANT, None, [255,255,255])
            images.append(model_image)
        
        output_path = '_concat_' + name + '.png'
        cv2.imwrite(output_path, cv2.hconcat(images))

        print('OK: ' + output_path)
