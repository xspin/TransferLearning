from model import Model, Config, default_config
import tensorflow as tf
import pickle
import numpy as np
import utils
from utils import config2str, Statistic, get_dataset, run, MyThread
import os, time, sys
import matplotlib.pyplot as plt
from config import config, param, param_name
# os.environ['TF_CPPMIN_LOG_LEVEL'] = '2'
tf.logging.set_verbosity(tf.logging.ERROR)


if __name__ == '__main__':

    # src_xdata, src_ydata, tgt_xdata, tgt_ydata = get_dataset(config)
    data = get_dataset(config)
    src_xdata, src_ydata, tgt_xdata, tgt_ydata = data
    src_name, tgt_name = config.src_name, config.tgt_name
    config = config._replace(input_shape=(src_xdata.shape[1],))

    print('='*80)
    print('Source Data: {} --> Target Data: {}'.format(src_name, tgt_name))
    print('src_xdata:', src_xdata.shape)
    print('src_ydata:', src_ydata.shape)
    print('tgt_xdata:', tgt_xdata.shape)
    print('tgt_ydata:', tgt_ydata.shape)
    print('='*80)

    accuracy = []
    clock_param = utils.Clock(len(param[param_name]))
    for i,v in enumerate(param[param_name]):
        cfg = config._replace(**{param_name:v})
        tf.reset_default_graph()
        with tf.Session() as sess:
            acc = run(sess, cfg, data)
        accuracy.append(acc)
        print('{}/{}  [{} = {}] acc_tgt: {} Cost: {} ETA: {}'.format(
            i+1, len(param[param_name]), param_name, v, acc, *clock_param.toc(i)))
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
        

                    