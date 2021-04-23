import numpy as np
import gzip
import _pickle as cPickle
import time

f = gzip.open('mnist.pkl.gz','rb')
data = cPickle.load(f, encoding='latin1')
f.close()
print(type(data), len(data))
train_set, valid_set, test_set = data

def resize_xs(xs, rate):
    N, dim = xs.shape
    if dim != 28*28:
        print('size wrong', dim)
        return
    nsize = int(28/rate)
    xs_new = np.zeros((N, nsize*nsize), dtype=np.float32)
    for i in range(N):
        xi = xs[i].reshape(28,28)
        xs_new[i] = xi[::rate,::rate].reshape(nsize*nsize)
    return xs_new

xs_tr, ys_tr = train_set
xs_te, ys_te = test_set
xs_val, ys_val = valid_set

# save
if True:
    for rate in [1, 2, 4, 7]:
        xs_tr_new = resize_xs(xs_tr, rate)
        xs_te_new = resize_xs(xs_te, rate)
        xs_val_new = resize_xs(xs_val, rate)
        print(xs_tr_new.shape, xs_te_new.shape, xs_val_new.shape)

        train_set_new = (xs_tr_new, ys_tr)
        valid_set_new = (xs_val_new, ys_val)
        test_set_new  = (xs_te_new, ys_te)
        nsize = int(28/rate)
        f2 = gzip.open('mnist_{}x{}.pkl.gz'.format(nsize, nsize),'wb')
        cPickle.dump((train_set_new, valid_set_new, test_set_new), f2)
        f2.close()
        print('resized images with rate={}'.format(rate))

def crop_down(xs, rate):
    N, dim = xs.shape
    if dim != 28*28:
        print('size wrong', dim)
        return
    nsize = int(20/rate)
    xs_new = np.zeros((N, nsize*nsize), dtype=np.float32)
    for i in range(N):
        xi = xs[i].reshape(28,28)
        xi = xi[4:24, 4:24]
        xs_new[i] = xi[::rate,::rate].reshape(nsize*nsize)
    return xs_new

if False:
    for rate in [1, 2]:
        xs_tr_new = crop_down(xs_tr, rate)
        xs_te_new = crop_down(xs_te, rate)
        xs_val_new = crop_down(xs_val, rate)
        print(xs_tr_new.shape, xs_te_new.shape, xs_val_new.shape)

        train_set_new = (xs_tr_new, ys_tr)
        valid_set_new = (xs_val_new, ys_val)
        test_set_new  = (xs_te_new, ys_te)
        nsize = int(20/rate)
        f2 = gzip.open('mnist_{}x{}.pkl.gz'.format(nsize, nsize),'wb')
        cPickle.dump((train_set_new, valid_set_new, test_set_new), f2)
        f2.close()
        print('resized images with size={}'.format(nsize))