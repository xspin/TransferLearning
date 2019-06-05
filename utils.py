import numpy as np
import matplotlib.pyplot as plt
import time
import tensorflow as tf
from sklearn.manifold import TSNE
import os
import pickle
import glob

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
            self.config.fe_shapes,
            self.config.classifier_shapes,
            self.config.discriminator_shapes
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
        self.ckpt_path = os.path.join(fig_dir, 'ckpt/model')

    def load(self, acc=None):
        if acc is None:
            dirs = glob.glob(os.path.join(self.path,'0.*'))
            dirs.sort(reverse=True)
            fig_dir = dirs[0]
        else:
            fig_dir = self.get_fig_dir(acc)
        pk_path = os.path.join(fig_dir, 'stat.pickle')
        if not os.path.exists(pk_path):
            print('#Warning: %s not exist'%pk_path)
        print('#Load data from', pk_path)
        with open(pk_path, 'rb') as f:
            self.stat = pickle.load(f)

        
if __name__ == '__main__':
    pass
    a = glob.glob('logs/books-electronics/0.*')
    print(a)
