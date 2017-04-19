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

import numpy as np
import os.path
import matplotlib.pyplot as plt
from mnist import load_mnist
import sys
import time

# ===============================================================================================================
# Training
# ===============================================================================================================

MODELNAME = 'baseline'

trainset_size = 5000
testset_size = 10000

# image and batch size
batch_size = 100
image_size = (1, 16, 16)

epoch = int(trainset_size/batch_size)  # 50=5000/100
max_iter = 100*epoch  # 5000
test_iter = 10
snapshot_iter = 500  # roughly 10 epochs

# Classifier
classifier = dict()
classifier['weights'] = ''
classifier['prototxt'] = './classifier/train.prototxt'
classifier['solver'] = './classifier/train.solver'

path2save = '/export/home/etikhonc/net/hci-storage01/userfolders/etikhonc/scratch/classification/MNIST_5000/' + MODELNAME + '/'
if not os.path.exists(path2save):
    os.mkdir(path2save)

# -------------------------------------------------------------------------------------------------------------------
def data_preprocess(dataset, datamean=None):
    dataset = dataset.astype('float32')
    if datamean is None:
        datamean = np.expand_dims(np.mean(dataset, axis=0), axis=0).astype('float32')
    dataset = dataset - np.tile(datamean, (dataset.shape[0], 1, 1))
    dataset /= 255.
    return dataset

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
    if start_snapshot > 0:
        classifier_caffemodel = path2save + 'classifier/' + '%s_%d.caffemodel' % (classifier['name'], start_snapshot)
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

        tstart = time.time()
        # create a new batch
        data = np.zeros(solver_C.net.blobs['data'].data.shape, dtype=np.float32)
        data[:, :, topleft[0]:topleft[0]+image_size[1], topleft[1]:topleft[1]+image_size[2]] = \
                        np.expand_dims(trainset[data_pointer:data_pointer+batch_size], axis=1)
        labels = trainlabels[data_pointer:data_pointer+batch_size]
        # feead the batch to the classifier
        solver_C.net.blobs['data'].data[...] = data
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
            testloss[it-start_snapshot] = testloss[it-1]
            testacc[it-start_snapshot] = testacc[it-1]
        f_testloss.write(str(it) + '\t' + str(testloss[it-start_snapshot, 0]) + '\n')
        f_testacc.write(str(it) + '\t' + str(testacc[it-start_snapshot, 0]) + '\n')

        # print out current state
        print 'epoch %3d/%3d\t loss=%.3f\t \t test_loss=%.3f\t t=%10.5fsec' %\
              (epoch_it, it, loss[it-start_snapshot, 0], testloss[it-start_snapshot, 0], tstop - tstart)

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
    parser.add_argument('-start', action='store', default=0, type=int)
    parser.add_argument('-gpu', action='store', default=0, type=int)
    arg = parser.parse_args()
    print '---------------------------------------------------'
    print 'GPU ID %d ' % arg.gpu
    print 'Continue training after %d iterations' % arg.start
    print '---------------------------------------------------'

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
