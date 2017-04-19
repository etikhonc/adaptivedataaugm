# Ekaterina Sutter, March 2017

'''
reproducing the experiment on the MNIST dataset from the paper

"Adaptive data augmentation for image classification"
by A.Fawzi, H.Samulowitz, D.Turaga, P.Frossard, ICIP 2016

baseline experiment

'''

import os
os.environ['GLOG_minloglevel'] = '2'  # suprress Caffe verbose prints

import argparse
import caffe
import cv2
import cupy

import numpy as np
import os.path
import matplotlib.pyplot as plt
from mnist import load_mnist
import sys
import time

from utils import data_preprocess, getRotationMatrix2D
from utils import warpAffine_gpu

# ===============================================================================================================
# Training
# ===============================================================================================================

MODELNAME = 'randaugm_finetune'

trainset_size = 5000
testset_size = 10000

# image and batch size
batch_size = 100
image_size = (1, 16, 16)

epoch = int(trainset_size/batch_size)  # 50=5000/100
max_iter = 100*epoch  # 5000
test_iter = 10
snapshot_iter = 500  # roughly 10 epochs

# Augmentation parameters
# L = 1.0 #0.25
# delta_L = 0.05
# interpolMode = cv2.INTER_LINEAR
interpolMode = cv2.INTER_CUBIC
# borderMode = cv2.BORDER_REFLECT
borderMode = cv2.BORDER_REPLICATE

param_trafo = dict()
param_trafo['rotmin'] = -30
param_trafo['rotmax'] = 30
param_trafo['smin'] = 0.8
param_trafo['smax'] = 1.2
# param_trafo['tmin'] = 0
# param_trafo['tmax'] = 0


# Classifier
classifier = dict()
classifier['weights'] = ''
classifier['prototxt'] = './classifier/train.prototxt'
classifier['solver'] = './classifier/train.solver'

path2save = '/net/hci-storage01/userfolders/etikhonc/scratch/classification/MNIST_5000/' + MODELNAME + '/'
if not os.path.exists(path2save):
    os.mkdir(path2save)

gpu = 0

# -------------------------------------------------------------------------------------------------------------------
def randaugm(data, p=0.3, param_trafo=param_trafo):
    N = data.shape[0]
    data_augm = np.zeros(data.shape, dtype=np.float32)

    transf_prob = np.random.rand(N)
    count_augmented = 0
    for i in xrange(N):
        im = data[i, 0]

        if transf_prob[i] < p:
            count_augmented += 1

            angle = np.random.randint(param_trafo['rotmin'], param_trafo['rotmax'])
            scale = np.random.uniform(param_trafo['smin'], param_trafo['smax'])
            tx = 0.0  # np.random.randint(param_trafo['tmin'], param_trafo['tmax'])
            ty = 0.0  # np.random.randint(param_trafo['tmin'], param_trafo['tmax'])

            # apply the augmentation
            # T = getRotationMatrix2D(angle, scale, (tx,ty), (28, 28))
            # im_warp = cv2.warpAffine(im, T, (im.shape[1], im.shape[0]),
            #                          flags=interpolMode, borderMode=borderMode)

            # apply the augmentation
            T =  cv2.getRotationMatrix2D((28/2., 28/2.), angle, scale) + [[0, 0, tx], [0, 0, ty]]
            im_warp = cv2.warpAffine(im, T, (im.shape[1], im.shape[0]),
                                     flags=interpolMode, borderMode=borderMode)
            data_augm[i, 0] = im_warp.astype(np.float32)
        else:
            data_augm[i, 0] = im.astype(np.float32)

    return data_augm, count_augmented


# -------------------------------------------------------------------------------------------------------------------
def train(trainset, trainlabels, testset, testlabels, start_snapshot=0):

    datareader_train = caffe.Net('./classifier/data.prototxt', caffe.TRAIN)
    datareader_test = caffe.Net('./classifier/data.prototxt', caffe.TEST)

    # classifier
    if not os.path.exists(path2save + 'classifier/'):
        os.mkdir(path2save + 'classifier/')
    solver_C = caffe.get_solver(classifier['solver'])
    # for k in solver_C.net.blobs.keys():
    #     print k, solver_C.net.blobs[k].data.shape

    # load from snapshot
    classifier_caffemodel = '/net/hci-storage01/userfolders/etikhonc/scratch/classification/MNIST_5000/'\
                            'baseline/classifier/classifier_%d.caffemodel' % start_snapshot
    if os.path.isfile(classifier_caffemodel):
        print '\n === Starting Classifier from the snapshot %d ===\n' % start_snapshot
        solver_C.net.copy_from(classifier_caffemodel)
    else:
        raise Exception('File %s does not exist' % classifier_caffemodel)

    f_loss = open(path2save + 'train_loss.txt', 'w')
    loss = np.zeros((max_iter-start_snapshot, 1), dtype=np.float32)

    f_testloss = open(path2save + 'test_loss.txt', 'w')
    testloss = np.zeros((max_iter-start_snapshot, 1), dtype=np.float32)

    f_testacc = open(path2save + 'test_acc.txt', 'w')
    testacc = np.zeros((max_iter-start_snapshot, 1), dtype=np.float32)

    data_pointer = 0
    epoch_it = int(start_snapshot/epoch)

    topleft = [int( (solver_C.net.blobs['data'].data.shape[2] - image_size[1])/2.),
               int( (solver_C.net.blobs['data'].data.shape[3] - image_size[2])/2.) ]
    for it in xrange(start_snapshot, max_iter):
        # end of one epoch
        if it > 0 and it % epoch == 0:
            print 'SHUFFLE DATA BEFORE STARTING A NEW EPOCH'
            data_pointer = 0
            epoch_it += 1
            ind = np.arange(trainset_size)
            np.random.shuffle(ind)
            trainset = trainset[ind]
            trainlabels = trainlabels[ind]

        # start timer

        # datareader_train.forward()
        # data = datareader_train.blobs['data'].data[...].copy()
        # labels = datareader_train.blobs['label'].data[...].copy().reshape(-1, 1)

        # start timer
        tstart = time.time()
        # create a new batch
        data = np.zeros(solver_C.net.blobs['data'].data.shape, dtype=np.float32)
        data[:, :, topleft[0]:topleft[0]+image_size[1], topleft[1]:topleft[1]+image_size[2]] = \
                        np.expand_dims(trainset[data_pointer:data_pointer+batch_size], axis=1)
        labels = trainlabels[data_pointer:data_pointer+batch_size]

        # apply random augmentation of the data
        data_randaugm, count_augmented = randaugm(data)

        # feead the batch to the classifier
        solver_C.net.blobs['data'].data[...] = data_randaugm
        solver_C.net.blobs['label'].data[...] = labels
        solver_C.step(1)
        # stop timer
        tstop = time.time()

        loss[it-start_snapshot, 0] = solver_C.net.blobs['loss'].data[...].copy()
        f_loss.write(str(it) + '\t' + str(loss[it-start_snapshot, 0]) + '\n')

        if it % test_iter == 0:
            for i in range(0, int(testset_size/batch_size)):
                # datareader_test.forward()
                # data_test = datareader_test.blobs['data_test'].data[...].copy()
                # labels_test = datareader_test.blobs['label_test'].data[...].copy().reshape(-1, 1)

                data_test = np.zeros(solver_C.test_nets[0].blobs['data'].data.shape, dtype=np.float32)
                data_test[:, :, topleft[0]:topleft[0]+image_size[1], topleft[1]:topleft[1]+image_size[2]] = \
                                np.expand_dims(testset[i*batch_size:(i+1)*batch_size], axis=1)
                labels_test = testlabels[i*batch_size:(i+1)*batch_size]

                solver_C.test_nets[0].blobs['data'].data[...] = data_test
                solver_C.test_nets[0].blobs['label'].data[...] = labels_test
                solver_C.test_nets[0].forward()
                testloss[it-start_snapshot] += solver_C.test_nets[0].blobs['loss'].data.copy()
                testacc[it-start_snapshot] += solver_C.test_nets[0].blobs['accuracy@1'].data.copy()
            testloss[it-start_snapshot] /= float(int(testset_size/batch_size))
            testacc[it-start_snapshot] /= float(int(testset_size/batch_size))
        else:
            testloss[it-start_snapshot] = testloss[it-start_snapshot-1]
            testacc[it-start_snapshot] = testacc[it-start_snapshot-1]
        f_testloss.write(str(it) + '\t' + str(testloss[it-start_snapshot, 0]) + '\n')
        f_testacc.write(str(it) + '\t' + str(testacc[it-start_snapshot, 0]) + '\n')

        # print out current state
        print 'epoch %3d/%3d\t %d/%d augmented \t loss=%.3f\t \t test_loss=%.3f\t t=%10.5fsec' %\
              (epoch_it, it, count_augmented, data_randaugm.shape[0],
              loss[it-start_snapshot, 0], testloss[it-start_snapshot, 0], tstop - tstart)

        if (it % snapshot_iter == 0 and it > start_snapshot):
            print ' SAVE weights'
            solver_C.net.save(os.path.join(path2save, 'classifier', 'classifier_%d.caffemodel' % it))

        data_pointer += batch_size

    f_loss.close()
    f_testloss.close()
    f_testacc.close()

    print ' SAVE weights'
    solver_C.net.save(os.path.join(path2save, 'classifier', 'classifier_%d.caffemodel' % it))

# -------------------------------------------------------------------------------------------------------------------


# -------------------------------------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='start parser')
    parser.add_argument('-start', action='store', default=40*epoch, type=int)
    parser.add_argument('-gpu', action='store', default=0, type=int)
    arg = parser.parse_args()
    print '---------------------------------------------------'
    print 'GPU ID %d ' % arg.gpu
    print 'Continue training after %d iterations' % arg.start
    print '---------------------------------------------------'
    gpu = arg.gpu

    caffe.set_device(arg.gpu)
    caffe.set_mode_gpu()

    # load dataset
    if not (os.path.isfile('mnist_5000_trainset.npy') and os.path.isfile('mnist_5000_trainlabels.npy')):
        trainset, trainlabels = load_mnist(dataset="training", imsize=image_size[1:], path="../MNIST")
        assert trainset.shape[0] == 60000
        trainmean = np.expand_dims(np.mean(trainset, axis=0), axis=0)
        plt.figure()
        plt.imshow(np.uint8(trainmean[0]))
        plt.axis('off')
        plt.savefig('trainmean.png', cmap='gray')
        plt.close('all')
        # random subset
        ind = np.random.choice(trainset.shape[0], trainset_size,replace=False)
        trainset = trainset[ind]
        trainlabels = trainlabels[ind]
        np.save('mnist_5000_trainset.npy', trainset)
        np.save('mnist_5000_trainlabels.npy', trainlabels)
        np.save('mnist_5000_trainmean.npy', trainmean)
    else:
        trainset = np.load('mnist_5000_trainset.npy')
        trainlabels = np.load('mnist_5000_trainlabels.npy')
        trainmean = np.load('mnist_5000_trainmean.npy')
    assert trainset.shape[0] == trainset_size, 'trainset has wrong number of samples: %d vs %d' % (trainset.shape[0], trainset_size)

    testset, testlabels = load_mnist(dataset="testing", imsize=image_size[1:], path="../MNIST")
    assert testset.shape[0] == testset_size, 'testset has wrong number of samples: %d vs %d' % (testset.shape[0], testset_size)
    testmean = np.expand_dims(np.mean(testset, axis=0), axis=0)

    trainset = data_preprocess(trainset, trainmean)
    testset = data_preprocess(testset, testmean)

    # train the network
    train(trainset, trainlabels, testset, testlabels, arg.start)

# -------------------------------------------------------------------------------------------------------------------


if __name__ == '__main__':
    main()
