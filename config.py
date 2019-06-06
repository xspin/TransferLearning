from model import Config
import os

config = Config(data_path = 'dataset/Amazon_review/amazon_acl.pickle',
                none = None,
                log_path = 'logs',
                # books dvd electronics kitchen
                src_name = 'books',
                tgt_name = 'electronics',
                epochs_step1 = 50,
                epochs_step2 = 700,
                batch_size = 64,
                n_classes = 2,
                input_shape = (5000,),
                gamma = 1, # loss_d in pretraining
                alpha = 10, # loss_mmd
                beta = 10, # loss_d
                lr_fd = 0.0001,
                lr_fg = 0.0001,
                drop_rate = 0.5,
                fe_shapes = [64], # the shapes of feature extractor 
                classifier_shapes = [64],
                discriminator_shapes = [64])
if os.name == 'posix': # just for debug
    config = config._replace(epochs_step1=3, epochs_step2=5)

param = dict()
param['gamma'] = [0.5, 1]
param['alpha'] = [1,2,3,4,5,6,7,8,9]
param['beta'] = [0.2, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9, 2.3, 2.5, 2.7, 3.2, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7]
param['lr_fg'] = [0.0001]
param['drop_rate'] = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
param['none'] = [None]

param_name = 'gamma'
