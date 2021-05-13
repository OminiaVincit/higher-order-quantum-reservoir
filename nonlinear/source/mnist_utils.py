import os
import numpy as np

# Generate MNIST dataset with appropriate size
def gen_mnist_dataset(mnist_dir, mnist_size):
    import gzip
    import _pickle as cPickle

    f = gzip.open(os.path.join(mnist_dir, 'mnist_{}.pkl.gz'.format(mnist_size)),'rb')
    data = cPickle.load(f, encoding='latin1')
    f.close()
    train_set, valid_set, test_set = data

    xs_train, ys_train = train_set
    xs_test, ys_test = test_set
    xs_val, ys_val = valid_set

    xs_train = xs_train / 255.0
    xs_test = xs_test / 255.0
    xs_val  = xs_val / 255.0

    return xs_train, ys_train, xs_test, ys_test, xs_val, ys_val

def get_acc(predict, out_lb):
    pred_lb = np.argmax(predict, axis=1)
    acc = np.sum(pred_lb == out_lb) / pred_lb.shape[0]
    return acc