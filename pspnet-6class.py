#! /usr/bin/env python3
# -*- coding: utf-8 -*-

########################################################################
## 
## PSPNet の Keras による実装 WIP
## 
## 参考:
## - https://arxiv.org/pdf/1612.01105.pdf
## - image-segmentation-keras
## - https://github.com/yutaroyamanaka/semantic-segmentation/blob/master/pspnet/model.py
##   https://tarovel4842.hatenablog.com/entry/2019/11/15/180322
## 
## imgaug でデータ拡張
## https://imgaug.readthedocs.io/en/latest/source/examples_segmentation_maps.html
## 
########################################################################

import os
import sys
import random
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # 使用するGPU番号

import numpy as np
import keras
from keras.models import *
from keras.layers import *
from keras.callbacks import *
import keras.backend as K

from tqdm import tqdm

import cv2
import itertools
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
ia.seed(1)


class PSPNetFactory:
    
    def __init__(self):
        self

    def conv_block(self, input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):

        filters1, filters2, filters3 = filters

        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = Conv2D(filters=filters1, kernel_size=(1, 1), strides=strides, name=conv_name_base + '2a')(input_tensor)
        x = BatchNormalization(axis=3, name=bn_name_base + '2a')(x)
        x = Activation(activation='relu')(x)

        x = Conv2D(filters=filters2, kernel_size=kernel_size, padding='same', name=conv_name_base + '2b')(x)
        x = BatchNormalization(axis=3, name=bn_name_base + '2b')(x)
        x = Activation(activation='relu')(x)

        x = Conv2D(filters=filters3, kernel_size=(1, 1), name=conv_name_base + '2c')(x)
        x = BatchNormalization(axis=3, name=bn_name_base + '2c')(x)

        shortcut = Conv2D(filters=filters3, kernel_size=(1, 1), strides=strides, name=conv_name_base + '1')(input_tensor)
        shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(shortcut)

        x = add(inputs=[x, shortcut])
        x = Activation(activation='relu')(x)
        return x

    def identity_block(self, input_tensor, kernel_size, filters, stage, block):

        filters1, filters2, filters3 = filters

        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = Conv2D(filters=filters1, kernel_size=(1, 1), name=conv_name_base + '2a')(input_tensor)
        x = BatchNormalization(axis=3, name=bn_name_base + '2a')(x)
        x = Activation(activation='relu')(x)

        x = Conv2D(filters=filters2, kernel_size=kernel_size, padding='same', name=conv_name_base + '2b')(x)
        x = BatchNormalization(axis=3, name=bn_name_base + '2b')(x)
        x = Activation(activation='relu')(x)

        x = Conv2D(filters=filters3, kernel_size=(1, 1), name=conv_name_base + '2c')(x)
        x = BatchNormalization(axis=3, name=bn_name_base + '2c')(x)

        x = add(inputs=[x, input_tensor])
        x = Activation(activation='relu')(x)
        return x

    def pool_block(self, feats, pool_factor):

        h = K.int_shape(feats)[1]
        w = K.int_shape(feats)[2]

        pool_size = strides = [
            int(np.round(float(h) / pool_factor)),
            int(np.round(float(w) / pool_factor))]

        x = AveragePooling2D(pool_size=pool_size, strides=strides, padding='same')(feats)
        x = Conv2D(filters=512, kernel_size=(1, 1), padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation(activation='relu')(x)

        x = self.resize_image(input=x, factors=strides)
        return x

    def resize_image(self, input, factors):
        return Lambda(lambda x: K.resize_images(x=x,
                                                height_factor=factors[0],
                                                width_factor=factors[1],
                                                data_format='channels_last',
                                                interpolation='bilinear'))(input)

    def pyramid_pooling_module(self, input, pool_factors=[1, 2, 3, 6]):
        pool_outs = [input]

        for p in pool_factors:
            pooled = self.pool_block(input, p)
            pool_outs.append(pooled)

        o = Concatenate(axis = -1)(pool_outs)
        return o

    def create(self, input_height, input_width, n_classes, with_auxiliary_loss=True):

        assert input_height % 32 == 0
        assert input_width % 32 == 0

        input_shape = (input_height, input_width, 3)

        input = Input(shape = input_shape)
        
        # build basic ResNet50 structure

        x = ZeroPadding2D(padding=(3, 3))(input)
        x = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), name='conv1')(x) # 1/2
        # f1 = x

        x = BatchNormalization(axis=3, name='bn_conv1')(x)
        x = Activation(activation='relu')(x)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x) # 1/4

        x = self.conv_block(x, kernel_size=3, filters=[64, 64, 256], stage=2, block='a', strides=(1, 1))
        x = self.identity_block(x, kernel_size=3, filters=[64, 64, 256], stage=2, block='b')
        x = self.identity_block(x, kernel_size=3, filters=[64, 64, 256], stage=2, block='c')
        # f2 = one_side_pad(x)

        x = self.conv_block(x, kernel_size=3, filters=[128, 128, 512], stage=3, block='a') # 1/8
        x = self.identity_block(x, kernel_size=3, filters=[128, 128, 512], stage=3, block='b')
        x = self.identity_block(x, kernel_size=3, filters=[128, 128, 512], stage=3, block='c')
        x = self.identity_block(x, kernel_size=3, filters=[128, 128, 512], stage=3, block='d')
        # f3 = x

        x = self.conv_block(x, kernel_size=3, filters=[128, 128, 512], stage=4, block='a', strides=(1, 1)) # 1/8
        x = self.identity_block(x, kernel_size=3, filters=[128, 128, 512], stage=4, block='b')
        x = self.identity_block(x, kernel_size=3, filters=[128, 128, 512], stage=4, block='c')
        x = self.identity_block(x, kernel_size=3, filters=[128, 128, 512], stage=4, block='d')
        x = self.identity_block(x, kernel_size=3, filters=[128, 128, 512], stage=4, block='e')
        x = self.identity_block(x, kernel_size=3, filters=[128, 128, 512], stage=4, block='f')
        f4 = x

        x = self.conv_block(x, kernel_size=3, filters=[128, 128, 512], stage=5, block='a', strides=(1, 1)) # 1/8
        x = self.identity_block(x, kernel_size=3, filters=[128, 128, 512], stage=5, block='b')
        x = self.identity_block(x, kernel_size=3, filters=[128, 128, 512], stage=5, block='c')
        f5 = x

        # Pyramid Pooling Module
        x = self.pyramid_pooling_module(input=f5, pool_factors=[1, 2, 3, 6])

        # main branch : Prediction Layers
        x = Conv2D(filters=512, kernel_size=(1, 1), use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation(activation='relu')(x)
        x = Conv2D(filters=n_classes, kernel_size=(3, 3), padding='same')(x)
        x = self.resize_image(input=x, factors=(8, 8)) # 1/1

        if with_auxiliary_loss:
            # sub branch for auxiliary loss feedback
            aux = f4
            # aux = self.pyramid_pooling_module(input=aux, pool_factors=[1, 2, 3, 6])
            # ここにも Pyramid Pooling Module いるのかなぁと思ったけど、この実装では入ってない。
            # https://github.com/hszhao/semseg/blob/master/model/pspnet.py
            aux = Conv2D(filters=256, kernel_size=(1, 1), use_bias=False)(aux)
            aux = BatchNormalization()(aux)
            aux = Activation(activation='relu')(aux)
            aux = Conv2D(filters=n_classes, kernel_size=(3, 3), padding='same')(aux)
            aux = self.resize_image(input=aux, factors=(8, 8)) # 1/1
        else:
            aux = None

        return self.get_segmentation_model(input=input, output=x, aux_output=aux)


    def get_segmentation_model(self, input, output, aux_output=None):

        o = output

        o_shape = Model(input, o).output_shape
        i_shape = Model(input, o).input_shape

        output_height = o_shape[1]
        output_width = o_shape[2]
        input_height = i_shape[1]
        input_width = i_shape[2]
        n_classes = o_shape[3]

        print("Input : {} x {}".format(input_height, input_width))
        print("Output : {} x {}".format(output_height, output_width))

        o = Reshape(target_shape=(output_height*output_width, -1))(o)
        o = Activation(activation='softmax')(o)

        if aux_output != None:
            aux_o_shape = Model(input, aux_output).output_shape
            assert output_height == aux_o_shape[1]
            assert output_width  == aux_o_shape[2]

            o_aux = Reshape(target_shape=(output_height*output_width, -1))(aux_output)
            o_aux = Activation(activation='softmax')(o_aux)
            model_output =[o, o_aux]
        else:
            model_output =[o]

        # for training
        model = Model(input, model_output)
        model.output_width = output_width
        model.output_height = output_height
        model.n_classes = n_classes
        model.input_height = input_height
        model.input_width = input_width
        model.model_name = "pspnet"

        return model

class Segmentation:

    def __init__(self, model):
        self.model = model

    def get_pairs_from_paths(self, images_path, segs_path):

        ACCEPTABLE_IMAGE_FORMATS = [".jpg", ".png"]
        ACCEPTABLE_SEGMENTATION_FORMATS = [".png"]

        image_files = []
        segmentation_files = {}

        for dir_entry in os.listdir(images_path):
            if os.path.isfile(os.path.join(images_path, dir_entry)) and \
                    os.path.splitext(dir_entry)[1] in ACCEPTABLE_IMAGE_FORMATS:
                file_name, file_extension = os.path.splitext(dir_entry)
                image_files.append((file_name, file_extension,
                                    os.path.join(images_path, dir_entry)))

        for dir_entry in os.listdir(segs_path):
            if os.path.isfile(os.path.join(segs_path, dir_entry)) and \
            os.path.splitext(dir_entry)[1] in ACCEPTABLE_SEGMENTATION_FORMATS:
                file_name, file_extension = os.path.splitext(dir_entry)
                full_dir_entry = os.path.join(segs_path, dir_entry)
                segmentation_files[file_name] = (file_extension, full_dir_entry)

        return_value = []
        for image_file, _, image_full_path in image_files:
            if image_file in segmentation_files:
                return_value.append((image_full_path,
                                    segmentation_files[image_file][1]))

        return return_value


    def image_segmentation_generator(self, images_path, segs_path, batch_size,
                                    n_classes, input_height, input_width,
                                    output_height, output_width, 
                                    augument_image=True, 
                                    with_auxiliary_loss=True):

        img_seg_pairs = self.get_pairs_from_paths(images_path, segs_path)
        random.shuffle(img_seg_pairs)
        zipped = itertools.cycle(img_seg_pairs)

        # https://imgaug.readthedocs.io/en/latest/source/api_augmenters_meta.html#imgaug.augmenters.meta.Sequential
        seq = iaa.Sequential([
            # https://imgaug.readthedocs.io/en/latest/source/overview/flip.html
            iaa.Fliplr(0.5),  # 水平反転を 50% の確率で適用
            iaa.Flipud(0.5),  # 垂直反転を 50% の確率で適用
            # https://imgaug.readthedocs.io/en/latest/source/overview/size.html#cropandpad
            iaa.Sometimes(0.5, iaa.CropAndPad(
                percent=(-0.1, 0.1), # 各辺ランダムに 10% 切り詰め（クロッピング） 〜 10% 埋め足し（パディング）
                pad_mode='constant', # パディング色は値指定
                pad_cval=0     # パディング色の値は 0（黒）
                # keep_size=False がないので元のサイズにリサイズされる
            )),
            # https://imgaug.readthedocs.io/en/latest/source/overview/geometric.html#affine
            # https://imgaug.readthedocs.io/en/latest/source/api_augmenters_geometric.html#imgaug.augmenters.geometric.Affine
            iaa.Sometimes(0.5, iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},               # 縦横に各 80% 〜 120% のリサイズ
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # 縦横に各 -20% 〜 20% の移動
                rotate=(-45, 45), # 回転角度は -45度 〜 +45度
                shear=(-16, 16),  # シアー角度は -16度 〜 +16度
                order=[0, 1],     # 補完方式は Nearest-neighbor か Bi-linear （ランダムに選択）
                cval=0,    # 背景色の値は 0（黒）
                mode='constant'   # 背景色は値指定
            )),
        ], random_order=True) # 適用順序はランダム

        while True:
            X = []
            Y = []
            for _ in range(batch_size):
                im, seg = next(zipped)

                im = cv2.imread(im, 1)
                seg = cv2.imread(seg, 1)

                if augument_image:
                    #  データ拡張
                    aug_det = seq.to_deterministic()
                    image_aug = aug_det.augment_image(im)
                    segmap = ia.SegmentationMapsOnImage(seg, shape=im.shape)
                    segmap_aug = aug_det.augment_segmentation_maps(segmap)
                    segmap_aug = segmap_aug.get_arr()
                    
                    im = image_aug
                    seg = segmap_aug

                X.append(self.get_image_array(im, input_width, input_height))
                Y.append(self.get_segmentation_array(seg, n_classes, output_width, output_height))

            if with_auxiliary_loss:
                yield np.array(X), [np.array(Y), np.array(Y)]
            else:
                yield np.array(X), [np.array(Y)]


    def get_image_array(self, image_input, width, height):

        if type(image_input) is np.ndarray:
            img = image_input
        else:
            img = cv2.imread(image_input, 1)

        img = cv2.resize(img, (width, height))
        img = img.astype(np.float32)
        img = img/255.0

        return img


    def get_segmentation_array(self, image_input, nClasses, width, height, no_reshape=False):
        """ Load segmentation array from input """

        seg_labels = np.zeros((height, width, nClasses))

        if type(image_input) is np.ndarray:
            img = image_input
        else:
            img = cv2.imread(image_input, 1)

        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_NEAREST)
        img = img[:, :, 0]

        for c in range(nClasses):
            seg_labels[:, :, c] = (img == c).astype(int)

        if not no_reshape:
            seg_labels = np.reshape(seg_labels, (width*height, nClasses))

        return seg_labels

    def predict(self, image_input):
        model = self.model

        output_width = model.output_width
        output_height = model.output_height
        input_width = model.input_width
        input_height = model.input_height
        n_classes = model.n_classes

        x = self.get_image_array(image_input=image_input, width=input_width, height=input_height)
        x = np.array([x])
        o = model.predict(x, batch_size=None, verbose=0, steps=None)
        
        if len(o) == 2:
            o = o[0]

        # print(o.shape) # -> (1, 331776, 7)
        result = o[0]
        # print(result.shape) # -> (331776, 7)

        pr = result.reshape((output_height,  output_width, n_classes))
        # print(pr.shape) # -> (576, 576, 7)

        return pr


    def evaluate(self, test_images, test_annotations, bootstrap_repeats=2000):
        model = self.model

        def mean_confidence_interval(a, confidence=0.95):
            n = len(a)
            m, se = np.mean(a), scipy.stats.sem(a)
            h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
            return m, m-h, m+h
        
        paths = self.get_pairs_from_paths(test_images, test_annotations)
        paths = list(zip(*paths))
        inp_images = list(paths[0])
        annotations = list(paths[1])

        print('== 各画像に対する IoU ==')
        print("data-id", end='')
        for i in range(model.n_classes) :
            print("\t{}_i\t{}_u\t{}_gt".format(i,i,i), end='')
        print()

        z = []
        for inp, ann, path in zip(inp_images, annotations, paths[0]):
            # 推論結果
            pr = self.predict(inp)
            pr = pr.argmax(axis=2)
            pr = pr.flatten()

            # 正解 ground truth
            gt = self.get_segmentation_array(image_input=ann, 
                                        nClasses=model.n_classes,
                                        width=model.output_width,
                                        height=model.output_height,
                                        no_reshape=True)
            gt = gt.argmax(-1)
            gt = gt.flatten()

            # 領域の計算（計算量が大きい部分）
            tp = np.zeros(model.n_classes) # true positive
            fp = np.zeros(model.n_classes) # false positive
            fn = np.zeros(model.n_classes) # false negative
            n_pixels = np.zeros(model.n_classes)
            for cl_i in range(model.n_classes):
                tp[cl_i] += np.sum((pr == cl_i) * (gt == cl_i))
                fp[cl_i] += np.sum((pr == cl_i) * ((gt != cl_i)))
                fn[cl_i] += np.sum((pr != cl_i) * ((gt == cl_i)))
                n_pixels[cl_i] += np.sum(gt == cl_i)
            z.append((tp, fp, fn, n_pixels))

            print(os.path.basename(path), end='')
            _union = tp + fp + fn
            for i in range(model.n_classes) :
                print("\t{:.0f}\t{:.0f}\t{:.0f}".format(tp[i], _union[i], n_pixels[i]), end='')
            print()
        print()

        print('== サンプルデータセットに対する IoU ==')
        tp = np.zeros(model.n_classes) # true positive
        fp = np.zeros(model.n_classes) # false positive
        fn = np.zeros(model.n_classes) # false negative
        n_pixels = np.zeros(model.n_classes)
        for _tp, _fp, _fn, _n_pixels in z:
            for i in range(model.n_classes):
                tp[i] += _tp[i]
                fp[i] += _fp[i]
                fn[i] += _fn[i]
                n_pixels[i] += _n_pixels[i]
        cl_wise_score = tp / (tp + fp + fn + 0.000000000001) # intersection over union
        n_pixels_norm = n_pixels / np.sum(n_pixels)
        frequency_weighted_IU = np.sum(cl_wise_score*n_pixels_norm)
        mean_IU = np.mean(cl_wise_score)
        
        for i in range(model.n_classes) :
            print("class_{}_IoU:\t{:.4f}".format(i, cl_wise_score[i]))
        print("mean_IoU:\t{:.4f}".format(mean_IU))
        print("frequency_weighted_IU:\t{:.4f}".format(frequency_weighted_IU))
        print()

        print('== サンプルデータセットに対する IoU のブートストラップ平均と 95% パーセンタイル区間 ==')
        _mean = []
        _freq = []
        _clsw = [[] for i in range(model.n_classes)]        
        for i in tqdm(np.arange(bootstrap_repeats)):
            
            tp = np.zeros(model.n_classes)
            fp = np.zeros(model.n_classes)
            fn = np.zeros(model.n_classes)
            n_pixels = np.zeros(model.n_classes)
            for idx in np.random.choice(len(z), len(z), replace=True):
                _tp, _fp, _fn, _n_pixels = z[idx]
                tp += _tp
                fp += _fp
                fn += _fn
                n_pixels += _n_pixels

            cl_wise_score = tp / (tp + fp + fn + 0.000000000001) # intersection over union
            n_pixels_norm = n_pixels / np.sum(n_pixels)
            frequency_weighted_IU = np.sum(cl_wise_score*n_pixels_norm)
            mean_IU = np.mean(cl_wise_score)

            _mean.append(mean_IU)
            _freq.append(frequency_weighted_IU)
            for i in range(model.n_classes) :
                _clsw[i].append(cl_wise_score[i])

        for i in range(model.n_classes) :
            _m, _l, _h = mean_confidence_interval(np.array(_clsw[i]))
            print("class_{}_IoU:\t{:.4f}\t{:.4f}\t{:.4f}".format(i, _m, _l, _h))
        _m, _l, _h = mean_confidence_interval(np.array(_mean))
        print("mean_IoU:\t{:.4f}\t{:.4f}\t{:.4f}".format(_m, _l, _h))
        _m, _l, _h = mean_confidence_interval(np.array(_freq))
        print("frequency_weighted_IU:\t{:.4f}\t{:.4f}\t{:.4f}".format(_m, _l, _h))


    def train(self,
            input_height,
            input_width,
            n_classes,
            train_images,
            train_annotations,
            val_images,
            val_annotations,
            checkpoints_path,
            epochs=5,
            batch_size=2,
            val_batch_size=2,            
            steps_per_epoch=512,
            val_steps_per_epoch=512,
            optimizer_name='adadelta',
            aux_loss_weight = 0.4):

        model = self.model

        output_height = model.output_height
        output_width = model.output_width

        if aux_loss_weight == None:
            with_auxiliary_loss = False
            model.compile(
                loss=['categorical_crossentropy'],
                optimizer=optimizer_name, 
                metrics=['accuracy'])
        else:
            assert aux_loss_weight < 1.0
            assert aux_loss_weight >= 0.0
            with_auxiliary_loss = True
            model.compile(
                loss=['categorical_crossentropy', 'categorical_crossentropy'], 
                loss_weights=[1.0 - aux_loss_weight, aux_loss_weight],
                optimizer=optimizer_name, 
                metrics=['accuracy'])

        train_gen = self.image_segmentation_generator(
            train_images, train_annotations,  batch_size,  n_classes,
            input_height, input_width, output_height, output_width, 
            augument_image=True,
            with_auxiliary_loss=with_auxiliary_loss)

        val_gen = self.image_segmentation_generator(
            val_images, val_annotations,  val_batch_size, n_classes, 
            input_height, input_width, output_height, output_width, 
            augument_image=False,
            with_auxiliary_loss=with_auxiliary_loss)

        callbacks = [
            ModelCheckpoint(filepath=checkpoints_path+'/model.{epoch:02d}-{val_loss:.2f}.hdf5')
        ]

        os.makedirs(checkpoints_path, exist_ok=True)

        model.fit_generator(generator=train_gen,
                            steps_per_epoch=steps_per_epoch,
                            validation_data=val_gen,
                            validation_steps=val_steps_per_epoch,
                            epochs=epochs, 
                            callbacks=callbacks,
                            use_multiprocessing=False)

import numpy as np
import scipy.stats




help = """
Usage: pspnet-6class.py train
Usage: pspnet-6class.py predict weights_file output_suffix
Usage: pspnet-6class.py evaluate weights_file bootstrap_repeats
"""


if __name__ == "__main__":

    args = sys.argv
    if len(args) < 2 : exit(help)

    input_height = 576
    input_width  = 576
    n_classes    = 7 # 6+1
    dataset_base_dir = "./dataset"

    batch_size   = 6
    epoches      = 500

    checkpoints_path    = 'checkpoints-8x-aux'
    with_auxiliary_loss = True #False
    aux_loss_weight     = 0.4 #None

    model = PSPNetFactory().create(
        input_height=input_height, 
        input_width=input_width, 
        n_classes=n_classes, 
        with_auxiliary_loss=with_auxiliary_loss)
    
    model.summary()

    command = args[1]
    if command == 'train' :

        if len(args) > 2:
            weights_file = args[2]
            model.load_weights(weights_file)

        segmentation = Segmentation(model)
        segmentation.train(
            input_height      = input_height, 
            input_width       = input_width, 
            n_classes         = n_classes,
            train_images      = dataset_base_dir + "/train_images/",
            train_annotations = dataset_base_dir + "/train_annotations/",
            val_images        = dataset_base_dir + "/val_images/",
            val_annotations   = dataset_base_dir + "/val_annotations/",
            checkpoints_path  = checkpoints_path,
            epochs            = epoches,
            batch_size        = batch_size,
            val_batch_size    = batch_size,
            steps_per_epoch     = 512,
            val_steps_per_epoch = 120 / batch_size, # number of test images = 120
            aux_loss_weight   = aux_loss_weight)

    elif command == 'evaluate' :

        if len(args) < 4 : exit(help)
        
        weights_file      = args[2]
        bootstrap_repeats = int(args[3])
        
        model.load_weights(weights_file)
        segmentation = Segmentation(model)
        print("for テストデータセット")
        segmentation.evaluate(
            test_images       = dataset_base_dir + '/test_images/', 
            test_annotations  = dataset_base_dir + '/test_annotations/',
            bootstrap_repeats = bootstrap_repeats)
        print()
        print("for バリデーションデータセット")
        segmentation.evaluate(
            test_images       = dataset_base_dir + '/val_images/', 
            test_annotations  = dataset_base_dir + '/val_annotations/',
            bootstrap_repeats = bootstrap_repeats)
        print()
        print("for トレーニングデータセット")
        segmentation.evaluate(
            test_images       = dataset_base_dir + '/train_images/', 
            test_annotations  = dataset_base_dir + '/train_annotations/',
            bootstrap_repeats = bootstrap_repeats)
        
    elif command == 'predict' :

        if len(args) < 4 : exit(help)

        weights_file  = args[2]
        output_suffix = args[3]

        model.load_weights(weights_file)
        segmentation = Segmentation(model)

        labels = {
            0: "Background",
            1: "Fibrocalcific plaque",
            2: "Fibrous cap atheroma / TCFA",
            3: "Healed erosion/rupture",
            4: "Intimal xanthoma",
            5: "Pathological intimal thickening",
            6: "Calcified nodule",
        }
        colors = {
            0: (0, 0, 0),
            1: (1, 000, 255), # 赤
            2: (2, 127, 255), # オレンジ
            3: (3, 255, 255), # 黄
            4: (4, 000, 127), # 暗い赤
            5: (5, 127, 127), # 暗い黄
            6: (6, 255, 127), # 黄緑
        }

        os.makedirs(os.path.join(dataset_base_dir, 'test_predict'), exist_ok=True)

        for name in os.listdir(os.path.join(dataset_base_dir, 'test_images')):
            image_input_path = os.path.join(dataset_base_dir, 'test_images', name)

            if os.path.isfile(image_input_path) != True:
                continue
            if os.path.splitext(name)[1] != '.png':
                continue

            print(image_input_path)

            pr = segmentation.predict(image_input=image_input_path)
            # print(pr.shape) # -> (576, 576, 5)
            pr_height = pr.shape[0]
            pr_width  = pr.shape[1]

            for i in range(n_classes):
                # v = pr[:,:,i]
                # print("{} : {:.6f} ({})".format(i, np.max(v), labels[i]))
                img = np.zeros((pr_height, pr_width, 3))
                b, g, r = colors[i]
                img[:,:,0] = pr[:,:,i] * b
                img[:,:,1] = pr[:,:,i] * g
                img[:,:,2] = pr[:,:,i] * r
                filename = os.path.join(dataset_base_dir, 'test_predict', name) + ".{}_{}.png".format(output_suffix, i)
                cv2.imwrite(filename, img)
                print("wrote : {}".format(filename))

            pr = pr.argmax(axis=2)
            # print(pr.shape) # -> (576, 576)

            img = np.zeros((pr_height, pr_width, 3))

            for i in range(n_classes):
                v = 100 * np.count_nonzero(pr == i) / (pr_height * pr_width)
                print("{} : {}%".format(i, v))

                b, g, r = colors[i]

                rCh = (pr == i) * r
                gCh = (pr == i) * g
                bCh = (pr == i) * b

                img[:, :, 2] += rCh[:, :]
                img[:, :, 1] += gCh[:, :]
                img[:, :, 0] += bCh[:, :]

            filename = os.path.join(dataset_base_dir, 'test_predict', name) + ".{}.png".format(output_suffix)
            cv2.imwrite(filename, img)
            print("wrote : {}".format(filename))

    else:
        exit(help)
