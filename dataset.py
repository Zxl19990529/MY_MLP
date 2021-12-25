import numpy as np
import os

def read_data(f='a2b4c3d6.csv'):
    data = []
    for line in open(f,'r').readlines():
        line = line.strip()
        x,y = line.split(',')
        data.append([float(x),float(y)])
    return data

def data_iterator(x, y, batch_size, shuffle=True):
    indx = list(range(len(x)))
    if shuffle:
        np.random.shuffle(indx)

    for start_idx in range(0, len(x), batch_size):
        end_idx = min(start_idx + batch_size, len(x)) # use the tail of the dataset
        yield x[indx[start_idx: end_idx]], y[indx[start_idx: end_idx]]

def load_mnist_2d(data_dir):
    fd = open(os.path.join(data_dir, 'train-images.idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trX = loaded[16:].reshape((60000, 28 * 28)).astype(float)

    fd = open(os.path.join(data_dir, 'train-labels.idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trY = loaded[8:].reshape((60000))

    fd = open(os.path.join(data_dir, 't10k-images.idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    teX = loaded[16:].reshape((10000, 28 * 28)).astype(float)

    fd = open(os.path.join(data_dir, 't10k-labels.idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    teY = loaded[8:].reshape((10000))

    trX = (trX - 128.0) / 255.0
    teX = (teX - 128.0) / 255.0

    return trX, teX, trY, teY