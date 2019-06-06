import numpy as np
import matplotlib.pyplot as plt
import time
import tensorflow as tf
from sklearn.manifold import TSNE
import os
import pickle
import glob

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
    plt.scatter(h_norm[:n_hs,0], h_norm[:n_hs,1], color=colors, marker='o', s=10)
    colors = list(map(lambda x: 'red' if x>0 else 'blue', yt))
    plt.scatter(h_norm[n_hs:,0], h_norm[n_hs:,1], color=colors, marker='^', s=10)
    plt.xticks([])
    plt.yticks([])
    print('Cost:', clock.toc())
    if save_path:
        print('Save TSNE fig to', save_path)
        plt.savefig(save_path)
    if show: plt.show()

if __name__ == '__main__':
    clock = Clock()
    n = 700
    d = 64
    hs = 1+np.random.randn(n, d)*0.5
    ht = np.random.randn(n, d)
    ys = np.random.randint(0, 2, [n,1])
    yt = np.random.randint(0, 2, [n,1])
    tsne_plot(hs, ht, ys, yt, 'logs/tsne.png')
    t = clock.toc()
    print('cost:', t)

