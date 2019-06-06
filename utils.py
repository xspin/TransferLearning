import numpy as np
import matplotlib.pyplot as plt
import time
import tensorflow as tf
from sklearn.manifold import TSNE
import os
import pickle
import glob
# from model import Model
import model
# print(model.Model)

def get_dataset(config):
    with open(config.data_path,'rb') as f:
        data = pickle.load(f)

    # books dvd electronics kitchen
    src_name = config.src_name
    tgt_name = config.tgt_name
    src_data = data[src_name]
    tgt_data = data[tgt_name]

    src_x_pos = src_data['positive']
    src_x_neg = src_data['negative']
    src_xdata = np.concatenate((src_x_pos[0].toarray(), src_x_neg[0].toarray()), axis=0)
    src_ydata = np.r_[src_x_pos[1], src_x_neg[1]]

    tgt_xdata, tgt_ydata = tgt_data['unlabeled']
    tgt_xdata = tgt_xdata.toarray()
    tgt_x_pos = tgt_data['positive']
    tgt_x_neg = tgt_data['negative']
    # test_xdata = np.concatenate((tgt_x_pos[0].toarray(), tgt_x_neg[0].toarray()), axis=0)
    # test_ydata = np.r_[tgt_x_pos[1], tgt_x_neg[1]]

    src_ydata = src_ydata.reshape([len(src_ydata), 1])
    tgt_ydata = tgt_ydata.reshape([len(tgt_ydata), 1])
    return src_xdata, src_ydata, tgt_xdata, tgt_ydata

def generator(xdata, ydata, batchsize):
    n = len(xdata)
    index = np.arange(n)
    while True:
        np.random.shuffle(index)
        xdata = xdata[index]
        ydata = ydata[index]
        for i in range(0, n, batchsize):
            if i+batchsize > n: break
            yield xdata[i:i+batchsize], ydata[i:i+batchsize]

def sec2hms(sec):
    t = sec
    s, t = t%60, t//60
    m, t = t%60, t//60
    h, d = t%24, t//24
    if d > 0: return "{:.0f}d {:.0f}h {:.0f}m {:.0f}s".format(d,h,m,s)
    if h > 0: return "{:.0f}h {:.0f}m {:.0f}s".format(h,m,s)
    if m > 0: return "{:.0f}m {:.0f}s".format(m,s)
    return "{:.02f}s".format(s)

class Clock:
    def __init__(self, n_steps=None):
        self.tic()
        self.n_steps = n_steps
    def tic(self):
        self.start_time = time.time()
    def toc(self, step=None):
        cost = time.time() - self.start_time
        if step is None or self.n_steps is None: 
            return sec2hms(cost)
        else:
            step += 1
            return sec2hms(cost), sec2hms(cost*(self.n_steps-step)/step)

class Marker:
    def __init__(self):
        self.best = 0
        self.prev = 0
    def update(self, v):
        marker = ' '
        if v >= self.best:
            self.best = v
            marker = "*"
        elif v > self.prev:
            marker = "+"
        elif v < self.prev:
            marker = "-"
        self.prev = v
        return marker

def config2str(config):
    return "gamma: {} alpha: {} beta: {} dr: {} lr_fd: {} lr_fg: {}"\
        " FE_shape: {} C_shape: {} D_shape: {}".format(
            config.gamma,
            config.alpha,
            config.beta,
            config.drop_rate,
            config.lr_fd,
            config.lr_fg,
            config.fe_shapes,
            config.classifier_shapes,
            config.discriminator_shapes
            )

class Statistic:
    def __init__(self, config):
        # os.makedirs(path, exist_ok=True)
        self.config = config
        # self.fig_dir = os.path.join(path
        self.path = os.path.join(config.log_path, "{}-{}".format(config.src_name, config.tgt_name))
        self.log_path = os.path.join(self.path, 'log.txt')
        self.summary_path = os.path.join(self.path, 'summary')
        self.stat = dict()
    def update(self, name, value):
        if name in self.stat:
            self.stat[name].append(value)
        else:
            self.stat[name] = [value]
    def get_fig_dir(self, acc=None):
        def func(a):
            return '.'.join(map(str,a))
        if acc is None:
            acc = max(self.stat['acc_tgt'])
        prefix = "{}_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(
            acc,
            self.config.gamma,
            self.config.alpha,
            self.config.beta,
            self.config.drop_rate,
            self.config.lr_fd,
            self.config.lr_fg,
            func(self.config.fe_shapes),
            func(self.config.classifier_shapes),
            func(self.config.discriminator_shapes)
            )
        return os.path.join(self.path, prefix)

    def save(self):
        fig_dir = self.get_fig_dir()
        if os.path.exists(fig_dir):
            print('#Warning: [{}] already exists'.format(fig_dir))
        os.makedirs(fig_dir, exist_ok=True)
        for name, values in self.stat.items():
            fig_path = os.path.join(fig_dir, name+'.png')
            plt.figure()
            plt.plot(values)
            plt.xlabel(name)
            plt.ylabel('acc' if 'acc' in name else 'loss')
            plt.savefig(fig_path)
            print('#Save {} to {}'.format(name, fig_path))
        pk_path = os.path.join(fig_dir, 'stat.pickle')
        print('#Dump data to', pk_path)
        with open(pk_path, 'wb') as f:
            pickle.dump(self.stat, f)
        self.ckpt_path = os.path.join(fig_dir, 'ckpt/model.ckpt')

    def load(self, acc=None):
        fig_dir = self.get_acc_dir(acc)
        pk_path = os.path.join(fig_dir, 'stat.pickle')
        if not os.path.exists(pk_path):
            print('#Warning: %s not exist'%pk_path)
        print('#Load data from', pk_path)
        with open(pk_path, 'rb') as f:
            self.stat = pickle.load(f)

    def get_acc_dir(self, acc=None):
        if acc is None:
            dirs = glob.glob(os.path.join(self.path,'0.*'))
            dirs.sort(reverse=True)
            fig_dir = dirs[0]
        else:
            fig_dir = self.get_fig_dir(acc)
        return fig_dir

    def get_ckpt_dir(self, acc=None):
        fig_dir = self.get_acc_dir(acc)
        self.ckpt_dir = os.path.join(fig_dir, 'ckpt')
        return self.ckpt_dir

        

        
def tsne_plot(hs, ht, ys, yt, save_path=None, show=False):
    clock = Clock()
    h = np.vstack([hs, ht])
    y = np.vstack([ys, yt])
    n_hs = hs.shape[0]
    tsne = TSNE(init='random')
    h_tsne = tsne.fit_transform(h)
    h_min, h_max = h_tsne.min(0), h_tsne.max(0)
    h_norm = (h_tsne - h_min) / (h_max - h_min)  # 归一化

    plt.figure(figsize=(6, 6))
    colors = list(map(lambda x: 'red' if x>0 else 'blue', ys))
    plt.scatter(h_norm[:n_hs,0], h_norm[:n_hs,1], color=colors, marker='o', s=15, alpha=0.6)
    colors = list(map(lambda x: 'purple' if x>0 else 'darkcyan', yt))
    plt.scatter(h_norm[n_hs:,0], h_norm[n_hs:,1], color=colors, marker='^', s=15, alpha=0.6)
    plt.xticks([])
    plt.yticks([])
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    print('Cost:', clock.toc())
    if save_path:
        print('Save TSNE figure to', save_path)
        plt.savefig(save_path)
    if show or not save_path: plt.show()

def run(sess, config, data):
    src_xdata, src_ydata, tgt_xdata, tgt_ydata = data
    src_gen = generator(src_xdata, src_ydata, config.batch_size)
    tgt_gen = generator(tgt_xdata, tgt_ydata, config.batch_size)

    # sess = tf.Session()
    stat = Statistic(config)
    md = model.Model(config)
    md.summary()
    init = tf.global_variables_initializer()
    sess.run(init)
    saver = tf.train.Saver() # after initializer
    writer = tf.summary.FileWriter(stat.summary_path, sess.graph)
    # Pre-training Step & Training Step
    # for step in ['Pretraining', 'Training']:
    accuracy_tgt = []
    global_step = 0
    for step in ['Pretraining', 'Training']:
        print('='*80)
        if step == 'Pretraining': 
            print('Step Pre-training ...')
            n_epochs = config.epochs_step1
            md.init()
        else:
            print('Step Training ...')
            n_epochs = config.epochs_step2
            md.init_fe3()
        clock_epoch = Clock(n_epochs)
        mk_epoch = Marker()
        k_epoch = 1
        for epoch in range(n_epochs):
            if epoch % k_epoch == 0:
                print('{} Epoch #{}/{}'.format(step, epoch+1, n_epochs))
            n_batchs = max(len(src_ydata), len(tgt_ydata))//config.batch_size
            clock_batch = Clock(n_batchs)
            mk_batch = Marker()
            # acc_d = 0.5
            for batch in range(n_batchs):
                batch_src_x, batch_src_y = next(src_gen)
                batch_tgt_x, batch_tgt_y = next(tgt_gen)
                feed_dict = {md.ph_src_input: batch_src_x, 
                            md.ph_y_true_src: batch_src_y, 
                            md.ph_tgt_input: batch_tgt_x,
                            md.ph_y_true_tgt: batch_tgt_y, 
                            md.ph_drop_rate: config.drop_rate
                            }
                if step == 'Pretraining':
                    md.init_pre_fg()
                    sess.run([md.fg_op_pre], feed_dict=feed_dict)
                    md.init_pre_fd()
                    sess.run([md.fd_op_pre], feed_dict=feed_dict)
                else:
                    md.init_train_fg()
                    sess.run([md.fg_op_train], feed_dict=feed_dict)
                    md.init_train_fd()
                    sess.run([md.fd_op_train], feed_dict=feed_dict)

                if batch %(n_batchs//5) == 0:
                # if False:
                    acc_src, acc_tgt, acc_d, yloss, fgloss, fdloss, mmdloss = sess.run(
                            [md.acc_src, md.acc_tgt, md.acc_d, 
                                md.loss_y, md.fg_loss_pre, md.fd_loss_pre, 
                                md.loss_mmd], 
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
            feed_dict = {md.ph_src_input: src_xdata, 
                        md.ph_y_true_src: src_ydata,
                        md.ph_tgt_input: tgt_xdata,
                        md.ph_y_true_tgt: tgt_ydata,
                        md.ph_drop_rate: 0
                        }
            output = sess.run(
                [md.acc_src, md.acc_tgt, md.acc_d, md.fd_loss_pre, md.loss_mmd], 
                feed_dict=feed_dict)
            acc_src, acc_tgt, acc_d, loss_fd, loss_mmd = output
            cost, eta = clock_epoch.toc(epoch)
            accuracy_tgt.append(acc_tgt)
            for name, value in zip(['acc_src', 'acc_tgt', 'acc_d', 'loss_fd', 'loss_mmd'], output):
                stat.update(name, value)
            if epoch % k_epoch == 0:
                print('  >> max_acc_tgt: [{:.04f}] acc_src: {:.04f} acc_tgt: {:.04f}{}  Cost: {} ETA: {}'.format(
                    max(accuracy_tgt), acc_src, acc_tgt, mk_epoch.update(acc_tgt), cost, eta))
            summary = sess.run(md.merged_summary, feed_dict=feed_dict)
            global_step += 1
            writer.add_summary(summary, global_step)
        print('Maximum acc_tgt: {}'.format(max(accuracy_tgt)))
    stat.save()
    print('#Save model to', stat.ckpt_path)
    saver.save(sess, stat.ckpt_path)
    # sess.close()
    return max(accuracy_tgt)

if __name__ == '__main__':
    n = 100
    d = 128
    # hs = 1+np.random.randn(n, d)*0.5
    # ht = np.random.randn(n, d)
    # ys = np.random.randint(0, 2, [n,1])
    # yt = np.random.randint(0, 2, [n,1])
    # tsne_plot(hs, ht, ys, yt)

