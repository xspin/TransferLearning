from model import Model, Config, default_config
import tensorflow as tf
import pickle
import numpy as np
import utils
from utils import config2str, Statistic, get_dataset
import os, time, sys
import matplotlib.pyplot as plt
# os.environ['TF_CPPMIN_LOG_LEVEL'] = '2'
tf.logging.set_verbosity(tf.logging.ERROR)


def run(sess, config):
    stat = Statistic(config)
    model = Model(config)
    model.summary()
    init = tf.global_variables_initializer()
    saver = tf.train.Saver() # after initializer
    writer = tf.summary.FileWriter(stat.summary_path, sess.graph)
    sess.run(init)
    # Pre-training Step & Training Step
    # for step in ['Pretraining', 'Training']:
    accuracy_tgt = []
    global_step = 0
    for step in ['Pretraining', 'Training']:
        print('='*80)
        if step == 'Pretraining': 
            print('Step Pre-training ...')
            n_epochs = config.epochs_step1
            model.init()
        else:
            print('Step Training ...')
            n_epochs = config.epochs_step2
            model.init_fe3()
        clock_epoch = utils.Clock(n_epochs)
        mk_epoch = utils.Marker()
        k_epoch = 1
        for epoch in range(n_epochs):
            if epoch % k_epoch == 0:
                print('{} Epoch #{}/{}'.format(step, epoch+1, n_epochs))
            n_batchs = max(len(src_ydata), len(tgt_ydata))//config.batch_size
            clock_batch = utils.Clock(n_batchs)
            mk_batch = utils.Marker()
            # acc_d = 0.5
            for batch in range(n_batchs):
                batch_src_x, batch_src_y = next(src_gen)
                batch_tgt_x, batch_tgt_y = next(tgt_gen)
                feed_dict = {model.ph_src_input: batch_src_x, 
                            model.ph_y_true_src: batch_src_y, 
                            model.ph_tgt_input: batch_tgt_x,
                            model.ph_y_true_tgt: batch_tgt_y, 
                            model.ph_drop_rate: config.drop_rate
                            }
                if step == 'Pretraining':
                    model.init_pre_fg()
                    sess.run([model.fg_op_pre], feed_dict=feed_dict)
                    model.init_pre_fd()
                    sess.run([model.fd_op_pre], feed_dict=feed_dict)
                else:
                    model.init_train_fg()
                    sess.run([model.fg_op_train], feed_dict=feed_dict)
                    model.init_train_fd()
                    sess.run([model.fd_op_train], feed_dict=feed_dict)

                if batch %(n_batchs//5) == 0:
                # if False:
                    acc_src, acc_tgt, acc_d, yloss, fgloss, fdloss, mmdloss = sess.run(
                            [model.acc_src, model.acc_tgt, model.acc_d, 
                                model.loss_y, model.fg_loss_pre, model.fd_loss_pre, 
                                model.loss_mmd], 
                            feed_dict=feed_dict)
                    cost, eta = clock_batch.toc(batch)
                    print('    Batch: {:02d}/{}  acc_src: {:.03f} acc_tgt: {:.03f}{} acc_d: {:.03f}'\
                            ' src_loss: {:.04f} fg_loss: {:.04f} fd_loss: {:.04f} mmd_loss: {:.04f}'\
                            '  Cost: {} ETA: {}'.format(
                                batch+1, n_batchs, 
                                acc_src, acc_tgt, mk_batch.update(acc_tgt), acc_d,
                                yloss, fgloss, fdloss, mmdloss,
                                cost, eta
                                ))
            # Lookup the accuracy of SRC and TGT data over the whole dataset
            feed_dict = {model.ph_src_input: src_xdata, 
                        model.ph_y_true_src: src_ydata,
                        model.ph_tgt_input: tgt_xdata,
                        model.ph_y_true_tgt: tgt_ydata,
                        model.ph_drop_rate: 0
                        }
            output = sess.run(
                [model.acc_src, model.acc_tgt, model.acc_d, model.fd_loss_pre, model.loss_mmd], 
                feed_dict=feed_dict)
            acc_src, acc_tgt, acc_d, loss_fd, loss_mmd = output
            cost, eta = clock_epoch.toc(epoch)
            accuracy_tgt.append(acc_tgt)
            for name, value in zip(['acc_src', 'acc_tgt', 'acc_d', 'loss_fd', 'loss_mmd'], output):
                stat.update(name, value)
            if epoch % k_epoch == 0:
                print('  >> max_acc_tgt: [{:.04f}] acc_src: {:.04f} acc_tgt: {:.04f}{}  Cost: {} ETA: {}'.format(
                    max(accuracy_tgt), acc_src, acc_tgt, mk_epoch.update(acc_tgt), cost, eta))
            summary = sess.run(model.merged_summary, feed_dict=feed_dict)
            global_step += 1
            writer.add_summary(summary, global_step)
        print('{}->{} Maximum acc_tgt: {}'.format(src_name, tgt_name, max(accuracy_tgt)))
    stat.save()
    print('#Save model to', stat.ckpt_path)
    saver.save(sess, stat.ckpt_path)
    return max(accuracy_tgt)

if __name__ == '__main__':

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
                    input_shape = (500,),
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

    src_xdata, src_ydata, tgt_xdata, tgt_ydata = get_dataset(config)
    src_name, tgt_name = config.src_name, config.tgt_name

    print('='*80)
    print('Source Data: {} --> Target Data: {}'.format(src_name, tgt_name))
    print('src_xdata:', src_xdata.shape)
    print('src_ydata:', src_ydata.shape)
    print('tgt_xdata:', tgt_xdata.shape)
    print('tgt_ydata:', tgt_ydata.shape)
    print('='*80)
    config = config._replace(input_shape=(src_xdata.shape[1],))

    src_gen = utils.generator(src_xdata, src_ydata, config.batch_size)
    tgt_gen = utils.generator(tgt_xdata, tgt_ydata, config.batch_size)


    param = dict()
    param['gamma'] = [0.5, 1, 1.5, 2]
    param['alpha'] = [1,2,3,4,5,6,7,8,9]
    param['beta'] = [0.2, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9, 2.3, 2.5, 2.7, 3.2, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7]
    param['lr_fg'] = [0.0001]
    param['drop_rate'] = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    param['none'] = [None]

    param_name = 'none'

    accuracy = []
    clock_param = utils.Clock(len(param[param_name]))
    for i,v in enumerate(param[param_name]):
        cfg = config._replace(**{param_name:v})
        with tf.Session() as sess:
            acc = run(sess, cfg)
        accuracy.append(acc)
        print('{}/{}  [{} = {}] Cost: {} ETA: {}'.format(
            i+1, len(param[param_name]), param_name, v, *clock_param.toc(i)))
    sess.close()
    stat = Statistic(config)
    print('#Saving logs to', stat.log_path)
    with open(stat.log_path,'a') as f:
        f.write("[{} {}] {} --> {}\n".format(time.time(), time.ctime(), src_name, tgt_name))
        f.write("{}\n".format(config2str(config)))
        f.write("{}: {}\n".format(param_name, str(param[param_name])))
        f.write("acc: {}\n\n".format(accuracy))

    if len(param[param_name])>1:
        print('#Plot figure ...')
        plt.figure()
        plt.plot(param[param_name], accuracy, 'b--.')
        plt.xlabel(param_name)
        plt.ylabel('acc')
        fig_path = os.path.join(stat.path, param_name+'.png')
        plt.savefig(fig_path)
    print('done')
    # plt.show()
        

                    