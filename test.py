# Ekaterina Sutter, February 2017
'''
baseline classification model trained on the 24 classes of the Imagenet-15
The classifier is caffenet reference model (caffe/models/bvlc_reference_caffenet/)
'''

import os
os.environ['GLOG_minloglevel'] = '2'  # suprress Caffe verbose prints

import argparse
import caffe

import matplotlib.pyplot as plt
from mnist import load_mnist
import numpy as np
import os.path

import time

# test set
testset_size = 10000

# image and batch size
batch_size = 100
image_size = (1, 16, 16)

# Classifier
classifier = dict()
classifier['weights'] = ''
classifier['prototxt'] = './classifier/test.prototxt'

path2save = '/net/hci-storage01/userfolders/etikhonc/scratch/classification/MNIST_5000/'

# -------------------------------------------------------------------------------------------------------------------
def data_preprocess(dataset, datamean=None):
    dataset = dataset.astype('float32')
    if datamean is None:
        datamean = np.expand_dims(np.mean(dataset, axis=0), axis=0).astype('float32')
    dataset = dataset - np.tile(datamean, (dataset.shape[0], 1, 1))
    dataset /= 255.
    return dataset

# -------------------------------------------------------------------------------------------------------------------
def test(testset, testlabels, path2snapshot):

    # classifier
    net = caffe.Net(classifier['prototxt'], path2snapshot, caffe.TEST)
    acc_top_1 = 0.
    acc_top_5 = 0.
    assert net.blobs['data'].data.shape[0]==batch_size

    topleft = [int( (net.blobs['data'].data.shape[2] - image_size[1])/2.),
               int( (net.blobs['data'].data.shape[3] - image_size[2])/2.) ]
    for i in range(0, int(testset_size/batch_size)):

        data_test = np.zeros(net.blobs['data'].data.shape, dtype=np.float32)
        data_test[:, :, topleft[0]:topleft[0]+image_size[1], topleft[1]:topleft[1]+image_size[2]] = \
                        np.expand_dims(testset[i*batch_size:(i+1)*batch_size], axis=1)
        labels_test = testlabels[i*batch_size:(i+1)*batch_size]

        net.blobs['data'].data[...] = data_test
        net.blobs['label'].data[...] = labels_test
        net.forward()

        acc_top_1 += net.blobs['accuracy@1'].data.copy()
        acc_top_5 += net.blobs['accuracy@5'].data.copy()

    acc_top_1 /= float(testset_size/batch_size)
    acc_top_5 /= float(testset_size/batch_size)

    print '--------------------------'
    print 'Top-1 accuracy: %0.2f -> test error: %0.2f' % (100 * acc_top_1, 100 - 100 * acc_top_1)
    print 'Top-5 accuracy: %0.2f' % (100 * acc_top_5)
    print '--------------------------'

    return
# -------------------------------------------------------------------------------------------------------------------


# -------------------------------------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='start parser')
    parser.add_argument('-gpu', action='store', default=0, type=int)
    parser.add_argument('-model', action='store', default='baseline', type=str)
    parser.add_argument('-snapshot', action='store', default=250, type=int)
    arg = parser.parse_args()
    print '---------------------------------------------------'
    print 'Test the classifier %s after %d iterations' % (arg.model, arg.snapshot)
    print '---------------------------------------------------'

    path2snapshot = os.path.join(path2save, arg.model, 'classifier/classifier_%d.caffemodel' % arg.snapshot)
    if not os.path.exists(path2snapshot):
        raise Exception('File %s doe not exists!!!' % path2snapshot)
        return

    caffe.set_device(arg.gpu)
    caffe.set_mode_gpu()

    testset, testlabels = load_mnist(dataset="testing", imsize=image_size[1:], path="../MNIST")
    assert testset.shape[0] == testset_size, 'testset has wrong number of samples: %d vs %d' % (testset.shape[0], testset_size)
    testmean = np.expand_dims(np.mean(testset, axis=0), axis=0)
    testset = data_preprocess(testset, testmean)

    test(testset, testlabels, path2snapshot)

# -------------------------------------------------------------------------------------------------------------------


if __name__ == '__main__':
    main()
