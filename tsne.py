from utils import Statistic, get_dataset, tsne_plot
from model import Model, Config 
import os, pickle
import tensorflow as tf
from config import config, param, param_name
tf.logging.set_verbosity(tf.logging.ERROR)

if __name__ == '__main__':

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
        tsne_plot(hs, ht, src_ydata, tgt_ydata, os.path.join(stat.get_acc_dir(), 'tsne.pdf'), show=True)


