import os, sys
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Model
import tensorflow.keras.layers as kl
import tensorflow.keras.models as km
import tensorflow.keras.callbacks as kc
import tensorflow.keras.backend as kb
import tensorflow.keras.regularizers as kr
import tensorflow.keras.optimizers as ko
from tensorflow.keras.utils import plot_model
from mmd.losses import mmd_loss, correlation_loss
# from utils import maximum_mean_discrepancy as mmd_loss
import utils
import config

def mlp(shapes, drop_rate, act='tanh', name='MLP'):
    model = km.Sequential(name=name)
    for i,n in enumerate(shapes):
            model.add(kl.Dense(n, activation=act, kernel_regularizer=kr.l2(0.01), name='{}-{}'.format(name, i)))
            # model.add(kl.Dropout(rate=1))
            model.add(kl.Lambda(lambda x:tf.nn.dropout(x, rate=drop_rate)))
    return model

def classifier(shapes, drop_rate, name='Classifier'):
    model = mlp(shapes, drop_rate, name=name)
    dense = kl.Dense(1, kernel_regularizer=kr.l2(0.01), activation='sigmoid', name='{}_output'.format(name))
    model.add(dense)
    return model

def discriminator(shapes, drop_rate, name='Discriminator'):
    model = mlp(shapes, drop_rate, name=name)
    dense = kl.Dense(1, kernel_regularizer=kr.l2(0.01), activation='sigmoid', name='{}_output'.format(name))
    model.add(dense)
    return model

def on_epoch_end(batch, logs, model, x_test, y_test, batch_size):
    score = model.evaluate(x_test, y_test, batch_size=batch_size)
    # print(model.metrics_names)
    test_acc.append(score[1])
    print('Test loss: {:.4f}  acc: {:.4f}, best acc: {:.4f}\n'.format(*score, max(test_acc)))

class Model:
    def __init__(self, config):
        # self.sess = tfsession
        self.fuck = 'shit'
        n_classes = 1 if config.n_classes == 2 else config.n_classes
        with tf.name_scope('Placeholders'):
            self.ph_src_input = tf.placeholder(tf.float32, shape=[None, *config.input_shape], name='SourcePH')
            self.ph_tgt_input = tf.placeholder(tf.float32, shape=[None, *config.input_shape], name='TargetPH')
            self.ph_y_true_src = tf.placeholder(tf.int32, shape=[None, n_classes], name='y_true_src')
            self.ph_y_true_tgt = tf.placeholder(tf.int32, shape=[None, n_classes], name='y_true_tgt')
            self.ph_drop_rate = tf.placeholder(tf.float32, name='drop_rate')
        # self.y_true_d = tf.placeholder(tf.int32, shape=[None, 1], name='y_true_d')

        with tf.name_scope('Inputs'):
            src_input = kl.Input(shape=config.input_shape, tensor=self.ph_src_input, name='Source')
            tgt_input = kl.Input(shape=config.input_shape, tensor=self.ph_tgt_input, name='Target')

        self.FE_src = mlp(config.fe_shapes, self.ph_drop_rate, name='FE1')
        self.FE_shared = mlp(config.fe_shapes, self.ph_drop_rate, name='FE2')
        self.FE_tgt = mlp(config.fe_shapes, self.ph_drop_rate, name='FE3')
        # self.FE_tgt = self.FE_src

        # with tf.name_scope('Concat'):
        src_feature_1 = self.FE_src(src_input)
        src_feature_2 = self.FE_shared(src_input)
        tgt_feature_2 = self.FE_shared(tgt_input)
        tgt_feature_1 = self.FE_tgt(tgt_input)
        self.src_feature = kl.Concatenate(name='Concat_src')([src_feature_1, src_feature_2])
        self.tgt_feature = kl.Concatenate(name='Concat_tgt')([tgt_feature_1, tgt_feature_2])

        self.Classifier = classifier(config.classifier_shapes, self.ph_drop_rate, name='Classifier')
        self.Discriminator = discriminator(config.discriminator_shapes, self.ph_drop_rate, name='Discriminator')

        y_dis_src = self.Discriminator(src_feature_2)
        y_dis_tgt = self.Discriminator(tgt_feature_2)

        # Outputs
        # with tf.name_scope('Outputs'):
        self.y_pred_src= self.Classifier(self.src_feature)
        self.y_pred_d = tf.concat([y_dis_src, y_dis_tgt], 0, name='pred_d')
        self.y_true_d = tf.concat([tf.zeros_like(y_dis_src, dtype=tf.int32), tf.ones_like(y_dis_tgt, dtype=tf.int32)], 0, name='true_d')
        self.y_pred_tgt = self.Classifier(self.tgt_feature)
        # print(self.y_pred_d.shape)

        # Metrics
        # with tf.name_scope('Accuracy'):
        self.acc_src = self.binary_acc(self.ph_y_true_src, self.y_pred_src)
        self.acc_d = self.binary_acc(self.y_true_d, self.y_pred_d)
        self.acc_tgt = self.binary_acc(self.ph_y_true_tgt, self.y_pred_tgt)

        # Losses
        # with tf.name_scope('Losses'):
        # self.loss_reg = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        self.loss_y = tf.losses.sigmoid_cross_entropy(
                                            multi_class_labels=self.ph_y_true_src, logits=self.y_pred_src)
        self.loss_d = tf.losses.sigmoid_cross_entropy(
                                            multi_class_labels=self.y_true_d, logits=self.y_pred_d)
        #self.loss_mmd = kl.Lambda(lambda x:mmd_loss(x[0],x[1],1), name='MMD_loss')([src_feature, tgt_feature])
        self.loss_mmd = kl.Lambda(lambda x:correlation_loss(x[0],x[1],1), name='MMD_loss')([self.src_feature, self.tgt_feature])

        # pretraining step
        self.fg_loss_pre = self.loss_y - config.gamma * self.loss_d
        self.fd_loss_pre = self.loss_d
        # training step
        self.fg_loss_train = self.loss_y + config.alpha*self.loss_mmd - config.beta*self.loss_d
        self.fd_loss_train = self.loss_d
        
        # Training operators
        self.fg_op_pre = tf.train.AdamOptimizer(config.lr_fg, name='Ada_fg_pre').minimize(self.fg_loss_pre)
        self.fd_op_pre = tf.train.AdamOptimizer(config.lr_fd, name='Ada_fd_pre').minimize(self.fd_loss_pre)
        self.fg_op_train = tf.train.AdamOptimizer(config.lr_fg, name='Ada_fg_train').minimize(self.fg_loss_train)
        self.fd_op_train = tf.train.AdamOptimizer(config.lr_fd, name='Ada_fd_train').minimize(self.fd_loss_train)

        # step 1
        self.model_step1 = km.Model([src_input, tgt_input], [self.y_pred_src, y_dis_src, y_dis_tgt])
        # step 2
        self.model_step2 = km.Model([src_input, tgt_input], [self.y_pred_src, y_dis_src, y_dis_tgt, self.loss_mmd])
        # step 3
        self.model_step3 = km.Model(tgt_input, self.y_pred_tgt)
        # overall model
        # self.model_whole = km.Model([src_input, tgt_input], 
        #                     [self.y_pred_src, y_dis_src, y_dis_tgt, self.loss_mmd])
    def summary(self):
        ''''
        summary of acc & loss:
            acc_src, acc_tgt, acc_d, loss_src, loss_d, loss_mmd
        '''
        tf.summary.scalar('acc_src', self.acc_src)
        tf.summary.scalar('acc_tgt', self.acc_tgt)
        tf.summary.scalar('acc_d', self.acc_d)
        tf.summary.scalar('loss_y', self.loss_y)
        tf.summary.scalar('loss_d', self.loss_d)
        tf.summary.scalar('loss_mmd', self.loss_mmd)
        # tf.summary.scalar('loss_reg', self.loss_reg)
        # tf.summary.histogram('FE_src', self.FE_src.get_weights()[0])
        # tf.summary.histogram('FE_tgt', self.FE_tgt.get_weights()[0])
        # tf.summary.histogram('FE_shared', self.FE_shared.get_weights()[0])
        # tf.summary.histogram('FE_src_avg', tf.reduce_mean(self.FE_src.get_weights()[0]))
        # tf.summary.histogram('FE_tgt_avg', tf.reduce_mean(self.FE_tgt.get_weights()[0]))
        # tf.summary.histogram('FE_shared_avg', tf.reduce_mean(self.FE_shared.get_weights()[0]))
        self.merged_summary = tf.summary.merge_all()

    def binary_acc(self, y_true, y_pred):
        y_pred_cmp = tf.math.greater(y_pred, tf.constant(0.5))
        y_pred_lb = tf.cast(y_pred_cmp, tf.int32)
        acc_cnt = tf.equal(y_pred_lb, y_true)
        acc = tf.reduce_mean(tf.cast(acc_cnt, tf.float32))
        return acc

    def get_models(self):
        return [self.model_step1, self.model_step2, self.model_step3]
    
    def plot_models(self):
        models = self.get_models()
        for i,m in enumerate(models):
            fn = 'model_%d.png'%(i+1)
            print("plot model-{} to {}".format(i+1, fn))
            plot_model(m, to_file=fn, rankdir='LR', show_shapes=True)

    def init_pre_fg(self):
        self.FE_src.trainable = True
        self.FE_shared.trainable = True
        self.Classifier.trainable = True
        self.Discriminator.trainable = False

    def init_pre_fd(self):
        self.FE_shared.trainable = False
        self.Discriminator.trainable = True

    def init_train_fg(self):
        self.FE_src.trainable = False
        self.FE_shared.trainable = True
        self.Classifier.trainable = True
        self.Discriminator.trainable = False

    def init_train_fd(self):
        self.FE_src.trainable = False
        self.FE_shared.trainable = False
        self.Discriminator.trainable = True

    def init(self):
        self.FE_src.trainable = True
        self.Classifier.trainable = True
        self.Discriminator.trainable = True

    def init_fe3(self):
        self.FE_tgt.set_weights(self.FE_src.get_weights())
        # self.FE_src.trainable = False
        # self.Classifier.trainable = False


if __name__ == '__main__':
    print(tf.__version__)
    model = Model(config.config)
    model.plot_models()
    # print(config2str(default_config))
    # print(np.mean(model.FE_shared.get_weights()[0], axis=1))
    # print((model.FE_shared.get_weights()[0].shape))