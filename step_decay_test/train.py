import sys
sys.path.append("../models")
sys.path.append("../")
import argparse
import math
import keras
import numpy as np
import tensorflow as tf
from keras import backend 
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import Callback, CSVLogger
from keras.models import Model
from keras import optimizers
from keras.layers import Input
from our_callbacks import TensorBoardWithLr, LearningRateScheduler, ModelCheckpointWithEpoch



if('tensorflow' == backend.backend()):
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

def main(args):

    learning_rate_scheduler = [[0.1, 0.01, 0.001], [0, 81, 122, 300]]
    img_rows      = 32
    img_cols      = 32
    img_channels  = 3
    num_classes   = 10
    mean          = []
    std           = []

    # load data
    # support cifar10, cifar100, fashion mnist
    # using meat/std data preprocessing method

    if args.data_set == "cifar10":
        from keras.datasets import cifar10 as DataSet
        num_classes = 10
        mean = [125.3, 123.0, 113.9]
        std  = [62.9932, 62.0887, 66.7048]
    elif args.data_set == "cifar100":
        from keras.datasets import cifar100 as DataSet
        num_classes = 100
        mean = [129.3, 124.1, 112.4]
        std  = [68.2, 65.4, 70.4]
    elif args.data_set =='fashion_mnist':
        from keras.datasets import fashion_mnist as DataSet
        num_classes = 10
        mean = [72.94042]
        std  = [90.02121]
    else:
        print("[ERROR] No data set %s " % args.data_set)
        exit()

    (x_train, y_train), (x_test, y_test) = DataSet.load_data()
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test  = keras.utils.to_categorical(y_test, num_classes)
    x_train = x_train.astype('float32')
    x_test  = x_test.astype('float32')

    epochs = args.first_lr_epoch_num + args.second_lr_epoch_num + args.third_lr_epoch_num;
    iterations  = (int)(math.ceil(len(x_train)*1. / args.batch_size))
    
    if args.batch_size <= 0:
    	print("[ERROR] batch size %d <= 0 " % args.data_set)
        exit()

    if args.data_set == 'fashion_mnist':
        img_rows     = 28
        img_cols     = 28
        img_channels = 1
        x_test = x_test.reshape((-1,img_rows,img_cols,img_channels))
        x_train = x_train.reshape((-1,img_rows,img_cols,img_channels))

    # do data preprocessing  [raw - mean / std]
    for i in range(len(mean)):
            x_train[:,:,:,i] = (x_train[:,:,:,i] - mean[i]) / std[i]
            x_test[:,:,:,i] = (x_test[:,:,:,i] - mean[i]) / std[i]

    # build network
    # support LeNet, ResNet, ResNeXt

    img_input = Input(shape=(img_rows,img_cols,img_channels))
    output = None
    if args.network == "lenet":
        from LeNet import lenet as NetWork
        learning_rate_scheduler[0] = [0.05, 0.005, 0.0005]
        learning_rate_scheduler[1] = [0, 81, 122, 300]
        output = NetWork().build(img_input,num_classes)
    elif args.network == "resnet":
        from ResNet import resnet as NetWork
        output = NetWork().build(img_input,num_classes, stack_n=args.network_depth)
        learning_rate_scheduler[0] = [0.1, 0.01, 0.001]
        learning_rate_scheduler[1] = [0, args.first_lr_epoch_num, args.second_lr_epoch_num+args.first_lr_epoch_num, 100000]
    elif args.network == "wresnet":
        from WResNet import wresnet as NetWork
        learning_rate_scheduler[0] = [0.1, 0.02, 0.004, 0.0008, 0.0001]
        learning_rate_scheduler[1] = [0, 60, 120, 160, 300]
        output = NetWork().build(img_input,num_classes, depth=args.network_depth, k=args.network_width)
    else:
        print("[ERROR] no network ", args.network)
        exit()

    model = Model(img_input, output)

    print(model.summary())

    # set optimizer
    # support SGD with momentum and Adam
    opt = None
    if args.optimizer == "sgd":
        opt = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
    elif args.optimizer == "adam":
        opt = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
    else:
        print("[ERROR] No optimizer ", args.optimizer)
        exit()

    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    # set callback
    tb_cb      = TensorBoardWithLr(log_dir=args.log_path, histogram_freq=0)
    change_lr  = LearningRateScheduler(args, iterations, learning_rate_scheduler)
    csv_logger = CSVLogger('./%s/training.csv' % args.log_path)
    ckpt       = ModelCheckpointWithEpoch("%s/weights.{epoch:02d}-{val_acc:.4f}.hdf5" % args.log_path, monitor='val_acc', save_best_only=True, save_begin_epoch=40)
    cbks       = [change_lr,tb_cb, csv_logger, ckpt]

    # using real-time data augmentation
    print('Using real-time data augmentation.')
    datagen = ImageDataGenerator(horizontal_flip=True,
                                 width_shift_range=0.125,
                                 height_shift_range=0.125,
                                 fill_mode='reflect')

    datagen.fit(x_train)

    # start traing 
    model.fit_generator(datagen.flow(x_train, y_train,batch_size=args.batch_size),
                        steps_per_epoch=iterations,
                        epochs=epochs,
                        callbacks=cbks,
                        validation_data=(x_test, y_test))
    # clear session
    backend.clear_session()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-b','--batch_size', type=int, default=128, metavar='NUMBER',
                    help='batch size(default: 128)')
    # parser.add_argument('-e','--epochs', type=int, default=200, metavar='NUMBER',
    #                 help='epochs(default: 200)')
    parser.add_argument('-o','--optimizer', type=str, default="sgd", metavar='STRING',
                    help='optimizer(default: sgd)')
    parser.add_argument('-d','--data_set', type=str, default="cifar10", metavar='STRING',
                    help='data set(default: cifar10)')
    parser.add_argument('-lr_m','--learning_rate_method', type=str, required=True, metavar='STRING',
                    help='learning rate method')
    parser.add_argument('-sc','--cosine_weight', type=float, default=0.5, metavar='FLOAT',
                    help='cosine weight of cosine-tanh combination(default: 0.5)')
    parser.add_argument('-net','--network', type=str, required=True, metavar='STRING',
                    help='network architecture')
    parser.add_argument('-log','--log_path', type=str, required=True, metavar='STRING',
                    help='log path')
    parser.add_argument('-depth','--network_depth', type=int, required=True, metavar='NUMBER',
                    help='the depth of network')
    parser.add_argument('-width','--network_width', type=int, default=1, metavar='NUMBER',
                    help='the width of WRN')
    parser.add_argument('-E1','--first_lr_epoch_num', type=int, required=True, metavar='NUMBER',
                    help='number of epoch to train on learning rate 0.1')
    parser.add_argument('-E2','--second_lr_epoch_num', type=int, required=True, metavar='NUMBER',
                    help='number of epoch to train on learning rate 0.01')
    parser.add_argument('-E3','--third_lr_epoch_num', type=int, required=True, metavar='NUMBER',
                    help='number of epoch to train on learning rate 0.001')
   
    args = parser.parse_args()
    print("\n=============== Argument ===============")
    print(args)

    main(args)
