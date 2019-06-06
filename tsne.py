from utils import Statistic, get_dataset, tsne_plot
from model import Model, Config 
import os, pickle
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

if __name__ == '__main__':

    config = Config(data_path = 'dataset/Amazon_review/amazon_acl.pickle',
                    none = None,
                    log_path = 'logs',
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
    print('='*80)
    print('Source Data: {} --> Target Data: {}'.format(config.src_name, config.tgt_name))
    print('src_xdata:', src_xdata.shape)
    print('src_ydata:', src_ydata.shape)
    print('tgt_xdata:', tgt_xdata.shape)
    print('tgt_ydata:', tgt_ydata.shape)
    print('='*80)
    config = config._replace(input_shape=(src_xdata.shape[1],))

    stat = Statistic(config)
    ckpt_dir = stat.get_ckpt_dir()

    tf.reset_default_graph()
    model = Model(config)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print('Restore model from', ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
        feed_dict = {model.ph_src_input: src_xdata, 
                    model.ph_y_true_src: src_ydata,
                    model.ph_tgt_input: tgt_xdata,
                    model.ph_y_true_tgt: tgt_ydata,
                    model.ph_drop_rate: 0
                    }
        # output = sess.run(
        #     [model.acc_src, model.acc_tgt, model.acc_d, model.fd_loss_pre, model.loss_mmd], 
        #     feed_dict=feed_dict)
        # acc_src, acc_tgt, acc_d, loss_fd, loss_mmd = output
        hs, ht = sess.run([model.src_feature, model.tgt_feature], feed_dict)
        print('shape of hs:', hs.shape)
        print('shape of ht:', ht.shape)
        tsne_plot(hs, ht, src_ydata, tgt_ydata, os.path.join(stat.get_acc_dir(), 'tsne.png'))


