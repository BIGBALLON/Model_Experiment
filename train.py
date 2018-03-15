import argparse
import math
import sys
import keras
import tensorflow as tf
from keras import backend 
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler, TensorBoard, Callback, CSVLogger
from keras.models import Model
from keras import optimizers
from keras.layers import Input

sys.path.append("./models")
if('tensorflow' == backend.backend()):
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

class TensorBoardWithLr(TensorBoard):
    def __init__(self, log_dir='./logs',
             histogram_freq=0,
             batch_size=32,
             write_graph=True,
             write_grads=False,
             write_images=False,
             embeddings_freq=0,
             embeddings_layer_names=None,
             embeddings_metadata=None):
        super(TensorBoardWithLr, self).__init__(log_dir,
                                             histogram_freq,
                                             batch_size,
                                             write_graph,
                                             write_grads,
                                             write_images,
                                             embeddings_freq,
                                             embeddings_layer_names,
                                             embeddings_metadata)


    def on_train_begin(self, logs=None):
        self.opt = self.model.optimizer
        self.opt_name = type(self.opt).__name__
        self.lr = self.opt.lr

    def on_batch_end(self, batch, logs=None):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = K.get_value(self.lr)
        summary_value.tag = 'real_lr'
        self.writer.add_summary(summary, K.get_value(self.opt.iterations))
        self.writer.flush()

    def on_epoch_end(self, epoch, logs=None):
        super(TensorBoardWithLr, self).on_epoch_end(epoch, logs)

num_classes   = 10
mean          = [125.3, 123.0, 113.9]
std           = [62.9932, 62.0887, 66.7048]
# older 
# learning_rate = [0.05, 0.01, 0.001, 0.0001]
# epoch_decay   = [0, 60, 120, 160, 300]
learning_rate = [0.1, 0.01, 0.001]
# epoch_decay   = [0, 100, 150, 300]
epoch_decay   = [0, 81, 122, 300]
start_lr      = learning_rate[0]
end_lr        = 0.0
method        = "tanh"
methods       = ['step_decay', 'tanh_restart', 'cos_restart', 'linear', 'abs_sin', 'exponential', 'tanh_tanh_restart', 'cos_tanh', 'tanh_epoch', 'cos_epoch', 'cos_restart_reducebytanh', 'tanh_restart_reducebytanh', 'cos_iteration', 'tanh_iteration']
log_path      = ""
cosine_weight = 0.0
epochs        = 0
iterations    = 0
batch_size    = 0

img_rows      = 32
img_cols      = 32
img_channels  = 3

class IterationLearningRateScheduler(Callback):
    def __init__(self):
        super(IterationLearningRateScheduler, self).__init__()
        if (method in methods) == False:
            print("[ERROR] no method ", method)
            exit()
        self.T_e    = 10.
        self.T_mul  = 2.
        self.T_next = self.T_e
        self.tt     = 0


    def on_train_begin(self, log=None):
        self.opt = self.model.optimizer

    def on_batch_end(self, batch, log):
        lr = K.get_value(self.opt.lr)
        iteration = K.get_value(self.opt.iterations)*1.
        if method == "linear":
            lr = start_lr + (end_lr-start_lr)*iteration/(iterations*epochs)

        elif method == "abs_sin":
            lr = start_lr*math.fabs( (iteration/(iterations*epochs) -1)*math.sin(2/((iteration/(iterations*epochs) -1))) )

        elif method == 'tanh_tanh_restart':
            dt = 1./(self.T_e*iterations)
            self.tt = self.tt+dt
            lr = (start_lr+end_lr)/2. - (start_lr-end_lr)/2. * math.tanh(8.*self.tt - 4.)
            lr = lr * (0.5 - 0.5 * math.tanh(8.*iteration/(iterations*epochs) - 4.))
            
        elif method == 'cos_tanh':
            cos = (start_lr+end_lr)/2.+(start_lr-end_lr)/2.*math.cos(math.pi/2.*(iteration/(iterations*epochs/2.))) 
            tanh = (start_lr+end_lr)/2. - (start_lr-end_lr)/2. * math.tanh(8.*iteration/(iterations*epochs) - 4.)
            lr = (cosine_weight*cos+(1-cosine_weight)*tanh)

        elif method == "cos_iteration":
            lr = (start_lr+end_lr)/2.+(start_lr-end_lr)/2.*math.cos(math.pi/2.*(iteration/(iterations*epochs/2.)))
        elif method == 'tanh_iteration':
            lr = (start_lr+end_lr)/2. - (start_lr-end_lr)/2. * math.tanh(8.*iteration/(iterations*epochs) - 4.)
        
        elif method == 'cos_restart':
            # cos without shift
            dt = math.pi/float(self.T_e)
            self.tt = self.tt+float(dt)/iterations
            if self.tt >= math.pi:
                self.tt = self.tt - math.pi
            lr = end_lr + 0.5*(start_lr - end_lr)*(1+ math.cos(self.tt))
        
        elif method == 'tanh_restart':
            # tanh restart
            dt = 1./(self.T_e*iterations)
            self.tt = self.tt+dt
            lr = (start_lr+end_lr)/2. - (start_lr-end_lr)/2. * math.tanh(8.*self.tt - 4.)
        
        # elif method == '1':
        #     # cos shift +pi/2
        #     dt = math.pi/float(self.T_e)
        #     self.tt = self.tt+float(dt)/iterations
        #     if self.tt >= math.pi:
        #         self.tt = self.tt - math.pi
        #     lr = end_lr +(start_lr - end_lr)*(1+ math.cos(self.tt + math.pi/2.))

        # elif method == '2':
        #     #  cos shift -pi/2
        #     # bas since the final learning rate would be large
        #     dt = math.pi/float(self.T_e)
        #     self.tt = self.tt+float(dt)/iterations
        #     if self.tt >= math.pi:
        #         self.tt = self.tt - math.pi
        #     lr = end_lr + (start_lr - end_lr)*(math.cos(self.tt - math.pi/2.))

        elif method == 'tanh_restart_reducebytanh':
            #  tanh as max and 0.1*tanh as min, multiply cos
            max_lr = (start_lr+end_lr)/2. - (start_lr-end_lr)/2. * math.tanh(8.*iteration/(iterations*epochs) - 4.)
            min_lr = 0.1*max_lr
            
            dt = math.pi/float(self.T_e)
            self.tt = self.tt+float(dt)/iterations
            if self.tt >= math.pi:
                self.tt = self.tt - math.pi
            lr = min_lr + (max_lr - min_lr)*(math.cos(self.tt - math.pi/2.))

        elif method == 'cos_restart_reducebytanh':
            #  tanh as max and 0.1*tanh as min, multiply tanh
            max_lr = (start_lr+end_lr)/2. - (start_lr-end_lr)/2. * math.tanh(8.*iteration/(iterations*epochs) - 4.)
            min_lr = 0.1*max_lr
            dt = 1./(self.T_e*iterations)
            self.tt = self.tt+dt
            lr = (max_lr+min_lr)/2. - (max_lr-min_lr)/2. * math.tanh(8.*self.tt - 4.)

        # elif method == '5':
        #     # cos shift +pi/2
        #     #  tanh as max and 0.1*tanh as min, multiply cos
        #     dt = math.pi/float(self.T_e)
        #     self.tt = self.tt+float(dt)/iterations
        #     if self.tt >= math.pi:
        #         self.tt = self.tt - math.pi
        #     max_lr = (start_lr+end_lr)/2. - (start_lr-end_lr)/2. * math.tanh(8.*iteration/(iterations*epochs) - 4.)
        #     min_lr = 0.1*max_lr
        #     lr = min_lr +(max_lr - min_lr)*(1+ math.cos(self.tt + math.pi/2.))
            
        # elif method == '6':
        #     #  cos shift -pi/2
        #     #  tanh as max and 0.1*tanh as min, multiply cos
        #     dt = math.pi/float(self.T_e)
        #     self.tt = self.tt+float(dt)/iterations
        #     if self.tt >= math.pi:
        #         self.tt = self.tt - math.pi
        #     max_lr = (start_lr+end_lr)/2. - (start_lr-end_lr)/2. * math.tanh(8.*iteration/(iterations*epochs) - 4.)
        #     min_lr = 0.1*max_lr
        #     lr = min_lr + (max_lr - min_lr)*(math.cos(self.tt - math.pi/2.))

        # elif method == '7':
        #     #  tanh restart
        #     #  cos as max and 0.1*tanh as min, multiply cos
        #     dt = 1./(self.T_e*iterations)
        #     self.tt = self.tt+dt
        #     max_lr = (start_lr+end_lr)/2.+(start_lr-end_lr)/2.*math.cos(math.pi/2.*(iteration/(iterations*epochs/2.)))
        #     min_lr = 0.1*((start_lr+end_lr)/2. - (start_lr-end_lr)/2. * math.tanh(8.*iteration/(iterations*epochs) - 4.))
        #     if max_lr < min_lr:
        #         tmp_lr = max_lr
        #         max_lr = min_lr
        #         min_lr = tmp_lr
        #     lr = (max_lr+min_lr)/2. - (max_lr-min_lr)/2. * math.tanh(8.*self.tt - 4.)
        # elif method == '8':
        #     #  cos shift pi/2
        #     #  cos as max and 0.1*tanh as min, multiply cos
        #     dt = math.pi/float(self.T_e)
        #     self.tt = self.tt+float(dt)/iterations
        #     if self.tt >= math.pi:
        #         self.tt = self.tt - math.pi
        #     max_lr = (start_lr+end_lr)/2.+(start_lr-end_lr)/2.*math.cos(math.pi/2.*(iteration/(iterations*epochs/2.)))
        #     min_lr = 0.1*((start_lr+end_lr)/2. - (start_lr-end_lr)/2. * math.tanh(8.*iteration/(iterations*epochs) - 4.))
        #     if max_lr < min_lr:
        #         tmp_lr = max_lr
        #         max_lr = min_lr
        #         min_lr = tmp_lr
        #     lr = min_lr + (max_lr - min_lr)*(1+math.cos(self.tt + math.pi/2.))
        # elif method == '9':
        #     #  cos shift -pi/2
        #     #  cos as max and 0.1*tanh as min, multiply cos
        #     dt = math.pi/float(self.T_e)
        #     self.tt = self.tt+float(dt)/iterations
        #     if self.tt >= math.pi:
        #         self.tt = self.tt - math.pi
        #     max_lr = (start_lr+end_lr)/2.+(start_lr-end_lr)/2.*math.cos(math.pi/2.*(iteration/(iterations*epochs/2.)))
        #     min_lr = 0.1*((start_lr+end_lr)/2. - (start_lr-end_lr)/2. * math.tanh(8.*iteration/(iterations*epochs) - 4.))
        #     if max_lr < min_lr:
        #         tmp_lr = max_lr
        #         max_lr = min_lr
        #         min_lr = tmp_lr
        #     lr = min_lr + (max_lr - min_lr)*(math.cos(self.tt - math.pi/2.))
    
        K.set_value(self.opt.lr, lr)

    def on_epoch_end(self, epoch, log):
    	lr = K.get_value(self.opt.lr)
        if method == "step_decay":
            for i in range(len(epoch_decay)):
                if epoch_decay[i] <= epoch < epoch_decay[i+1]:
                	lr = learning_rate[i]
        elif method == "exponential":
        	lr = start_lr*(0.98**epoch)
        elif method == "cos_epoch":
            lr = (start_lr+end_lr)/2.+(start_lr-end_lr)/2.*math.cos(math.pi/2.*(epoch/(epochs/2.)))
        elif method == 'tanh_epoch':
            lr = (start_lr+end_lr)/2. - (start_lr-end_lr)/2. * math.tanh(8.*epoch/epochs - 4.)
        K.set_value(self.opt.lr, lr)

        if(epoch+1 == self.T_next):
            self.tt = 0
            self.T_e = self.T_e*self.T_mul
            self.T_next = self.T_next + self.T_e

def main(args):
    
    global batch_size
    global epochs
    global iterations
    global method
    global cosine_weight
    global log_path
    global num_classes
    global epoch_decay
    global learning_rate
    global mean
    global std

    # load data
    data_set = None
    if args.data_set == "cifar10":
        from keras.datasets import cifar10 as DataSet
        num_classes = 10
    elif args.data_set == "cifar100":
        from keras.datasets import cifar100 as DataSet
        num_classes = 100
        mean = [129.3, 124.1, 112.4]
        std  = [68.2, 65.4, 70.4]
    else:
        print("[ERROR] No data set " , args.data_set)
        exit()

    (x_train, y_train), (x_test, y_test) = DataSet.load_data()
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test  = keras.utils.to_categorical(y_test, num_classes)
    x_train = x_train.astype('float32')
    x_test  = x_test.astype('float32')



    batch_size    = args.batch_size
    epochs        = args.epochs
    iterations    = (int)(math.ceil(len(x_train)*1. / batch_size))
    method        = args.learning_rate_method
    cosine_weight = args.cosine_constant
    log_path      = args.log_path
    
    

    if batch_size <= 0:
        print("[ERROR] batch size cannot be %d" % batch_size)
        exit()
    if epochs <= 0:
        print("[ERROR] epochs cannot be %d" % epochs)
        exit()


    # data preprocessing  [raw - mean / std]
    for i in range(3):
        x_train[:,:,:,i] = (x_train[:,:,:,i] - mean[i]) / std[i]
        x_test[:,:,:,i] = (x_test[:,:,:,i] - mean[i]) / std[i]

    # build network
    img_input = Input(shape=(img_rows,img_cols,img_channels))
    output = None
    if args.network == "resnet":
        from ResNet import resnet as NetWork
        output = NetWork(img_input,num_classes)

    elif args.network == "wresnet":
        from WResNet import wresnet as NetWork
        learning_rate = [0.1, 0.02, 0.004, 0.0008, 0.0001]
        epoch_decay   = [0, 60, 120, 160, 300]
        output = NetWork().build(img_input,num_classes)

    elif args.network == "senet":
        from SENet import senet as NetWork
        output = NetWork(img_input,num_classes)

    else:
        print("[ERROR] no network ", args.network)
        exit()

    # use when the epochs are not 200
    # adjust the decay timing 
    for i in range(len(epoch_decay)):
        epoch_decay[i] = epoch_decay[i]*(epochs*1./200)

    model = Model(img_input, output)

    print(model.summary())

    # set optimizer
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
    tb_cb = TensorBoardWithLr(log_dir=log_path, histogram_freq=0)
    change_lr = IterationLearningRateScheduler()
    csv_logger = CSVLogger('./%s/training.csv' % log_path)

    if args.optimizer == 'adam':
        cbks = [tb_cb, csv_logger]
    else:
        cbks = [change_lr,tb_cb, csv_logger]

    # using real-time data augmentation
    print('Using real-time data augmentation.')
    datagen = ImageDataGenerator(horizontal_flip=True,
            width_shift_range=0.125,height_shift_range=0.125,fill_mode='reflect')

    datagen.fit(x_train)

    # start traing 
    model.fit_generator(datagen.flow(x_train, y_train,batch_size=batch_size),
                        steps_per_epoch=iterations,
                        epochs=epochs,
                        callbacks=cbks,
                        validation_data=(x_test, y_test))
    # save model
    model.save('%s/weights.h5' % args.log_path)


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
    parser.add_argument('-sgdr_cur','--sgdr_curve', type=str, metavar='STRING',
                    help='the type of curve when using sgdr')
    parser.add_argument('-sc','--cosine_constant', type=float, default=0.5, metavar='FLOAT',
                    help='cosine weight of cosine-tanh combination(default: 0.5)')
    parser.add_argument('-net','--network', type=str, required=True, metavar='STRING',
                    help='network architecture')
    parser.add_argument('-log','--log_path', type=str, required=True, metavar='STRING',
                    help='log path')

    
    args = parser.parse_args()
    print(args)

    main(args)
