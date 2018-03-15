import argparse
import keras
import numpy as np
import math
from keras import optimizers
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from keras.callbacks import LearningRateScheduler, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend 

if('tensorflow' == backend.backend()):
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

num_classes   = 10
weight_decay  = 0.0001
mean          = [125.307, 122.95, 113.865]
std           = [62.9932, 62.0887, 66.7048]
learning_rate = [0.05, 0.01, 0.001, 0.0001]
epoch_decay   = [0, 60, 120, 160, 300]
start_lr      = learning_rate[0]
end_lr        = learning_rate[-1]
method        = "tanh"
cosine_weight = 0.0
epochs        = 0
iterations    = 0
batch_size    = 0
log_path  = ''

def build_model(args):
    model = Sequential()
    model.add(Conv2D(6, (5, 5), padding='valid', activation = 'relu', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay), input_shape=(32,32,3)))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Conv2D(16, (5, 5), padding='valid', activation = 'relu', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay)))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(120, activation = 'relu', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay) ))
    model.add(Dense(84, activation = 'relu', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay) ))
    model.add(Dense(10, activation = 'softmax', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay) ))
    opt = None
    if args.optimizer == "sgd":
        opt = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
    elif args.optimizer == "adam":
        opt = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)

    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

def scheduler(epoch):
    if method == "step_decay":
        for i in range(len(epoch_decay)):
            if epoch_decay[i] <= epoch < epoch_decay[i+1]:
                return learning_rate[i]
    elif method == "cos":
        return (start_lr+end_lr)/2+(start_lr-end_lr)/2*math.cos(math.pi/2*(epoch/(epochs/2)))
    elif method == "tanh":
        return (start_lr+end_lr)/2 - (start_lr-end_lr)/2 * math.tanh(8*epoch/epochs - 4)
    elif method == "linear":
        return start_lr + (end_lr-start_lr)*epoch/epochs
    elif method == "exponential":
        return start_lr*(0.98**epoch)
    elif method == "cos_tanh":
        return (cosine_weight * (start_lr+end_lr)/2+(start_lr-end_lr)/2*math.cos(math.pi/2*(epoch/(epochs/2))) +
         (1-cosine_weight)*(start_lr+end_lr)/2 - (start_lr-end_lr)/2 * math.tanh(8*epoch/epochs - 4))

def main(args):

    # load data
    data_set = None
    if args.data_set == "cifar10":
        from keras.datasets import cifar10 as DataSet
    elif args.data_set == "cifar100":
        from keras.datasets import cifar10 as DataSet

    (x_train, y_train), (x_test, y_test) = DataSet.load_data()
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test  = keras.utils.to_categorical(y_test, num_classes)
    x_train = x_train.astype('float32')
    x_test  = x_test.astype('float32')

    global batch_size
    global epochs
    global iterations
    global method
    global cosine_weight
    global log_path

    batch_size    = args.batch_size
    epochs        = args.epochs
    iterations    = len(x_train) // batch_size
    method        = args.learning_rate_method
    cosine_weight = args.cosine_constant
    log_path      = args.log_path

    # data preprocessing  [raw - mean / std]
    for i in range(3):
        x_train[:,:,:,i] = (x_train[:,:,:,i] - mean[i]) / std[i]
        x_test[:,:,:,i] = (x_test[:,:,:,i] - mean[i]) / std[i]

    # build network
    model = build_model(args)
    print(model.summary())

    # set callback
    tb_cb = TensorBoard(log_dir=log_path, histogram_freq=0)
    change_lr = LearningRateScheduler(scheduler)
    cbks = [change_lr,tb_cb]

    # using real-time data augmentation
    print('Using real-time data augmentation.')
    datagen = ImageDataGenerator(horizontal_flip=True,
            width_shift_range=0.125,height_shift_range=0.125,fill_mode='constant',cval=0.)

    datagen.fit(x_train)

    # start traing 
    model.fit_generator(datagen.flow(x_train, y_train,batch_size=batch_size),
                        steps_per_epoch=iterations,
                        epochs=epochs,
                        callbacks=cbks,
                        validation_data=(x_test, y_test))
    # save model
    model.save('lenet.h5')

if __name__ == '__main__':    

    parser = argparse.ArgumentParser()
    parser.add_argument('-b','--batch_size', type=int, default=128, metavar='NUMBER',
                    help='batch size(default: 128)')
    parser.add_argument('-e','--epochs', type=int, default=200, metavar='NUMBER',
                    help='epochs(default: 200)')
    parser.add_argument('-o','--optimizer', type=str, default="sgd", metavar='STRING',
                    help='optimizer(default: sgd)')
    parser.add_argument('-d','--data_set', type=str, default="cifar10", metavar='STRING',
                    help='data set(default: cifar10)')
    parser.add_argument('-lr_m','--learning_rate_method', type=str, required=True, metavar='STRING',
                    help='learning rate method')
    parser.add_argument('-sc','--cosine_constant', type=float, default=0.5, metavar='FLOAT',
                    help='cosine weight of cosine-tanh combination(default: 0.5)')
    parser.add_argument('-net','--network', type=str, required=True, metavar='STRING',
                    help='network architecture')
    parser.add_argument('-log','--log_path', type=str, required=True, metavar='STRING',
                    help='log path')
    # parser.add_argument('-lrn','--learning_rate_number', type=int, required=True, default=1, metavar='NUMBER',
    #                 help='learning rate number(default: 1)')
    # parser.add_argument('-lr1','--1st_learning_rate', type=float, default=0.1, metavar='FLOAT',
    #                 help='first learning rate(default: 0.1)')
    # parser.add_argument('-lr2','--2nd_learning_rate', type=float, default=0.01, metavar='FLOAT',
    #                 help='second learning rate(default: 0.01)')
    # parser.add_argument('-lr3','--3rd_learning_rate', type=float, default=0.001, metavar='FLOAT',
    #                 help='third learning rate(default: 0.001)')
    # parser.add_argument('-lr4','--4th_learning_rate', type=float, default=0.0001, metavar='FLOAT',
    #                 help='forth learning rate(default: 0.0001)')
    
    args = parser.parse_args()

    main(args)