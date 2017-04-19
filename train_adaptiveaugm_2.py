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
from skimage.transform import warp, AffineTransform
import sys
import time
from utils import data_preprocess, getRotationMatrix2D
from utils import constr_theta11_theta22, constr_theta12_theta_21
from utils import constr_theta13, constr_theta23
from utils import warpAffine_gpu


path2cplex = '/server/opt/cplex/CPLEX_Studio/cplex/'
path2concert = '/server/opt/cplex/CPLEX_Studio/concert/'

# path2pythonlibs = '/export/home/' + param['user'] + '/.local/lib/python2.7/site-packages/'
# sys.path.insert(1,  path2pythonlibs)

# from pycpx import CPlexModel
import cplex
from cplex.exceptions import CplexError

# ===============================================================================================================
# Training
# ===============================================================================================================

MODELNAME = 'adaptiveaugm_2'

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
L = 0.25 # theta is defined for an image with size 1x1 therefor factor 28
K = 5

# interpolMode = cv2.INTER_LINEAR
interpolMode = cv2.INTER_CUBIC
# borderMode = cv2.BORDER_REFLECT
borderMode = cv2.BORDER_REPLICATE

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
def optimize_param(net, im, imID, label, K=5, debug=False):

    C, H, W = im.shape

    # calculate Jacobian of the transformation: T(x,y,theta) = [[x*theta1 + y*theta2 + theta3],
    #                                                           [x*theta4 + y*theta5 + theta6]]
    J = np.zeros((H, W, 2, 6), dtype=np.float32)  # independent from the color channel
    for x in xrange(W):
        for y in xrange(H):
            J[y, x, ...] = np.asarray([[x - W/2., y + H/2., 1, 0, 0, 0],
                                       [0, 0, 0, x - W/2., y - H/2., 1]])

    # use the same starting poin for all images: zero transformation
    theta0 = np.reshape(np.asarray([1., 0, 0,
                                    0, 1., 0]), (6, 1))

    if debug:
        print '============================================='
        print 'Starting Transformation: ', theta0.reshape((-1))
        print '-------------------------------------------'
        print

    I = im[0]

    theta = np.zeros((6,K+1), dtype=np.float32)
    theta[:,0] = theta0[:, 0]

    energy_vec = 100 * np.ones((K+1), dtype=np.float32)
    I_vec = np.zeros((K+1, H,W), dtype=np.float32)

    # for k in xrange(K):
    k = 0
    while True:
        # warp image
        T = theta[:,k].copy().reshape(2,3)
        T[:,2] = H * T[:,2]  # theta is defined for an image with size 1x1 therefor factor 28
        I = cv2.warpAffine(I, T, (W, H), flags=interpolMode, borderMode=borderMode)
        # I = warpAffine_gpu(I, T, (H,W), gpuID=gpu)
        I_vec[k] = I

        # I = cv2.GaussianBlur(I, ksize=(3,3), sigmaX=0, sigmaY=0, borderType=borderMode)
        I_dx = cv2.Sobel(I, cv2.CV_64F, 1, 0, ksize=3)
        I_dy = cv2.Sobel(I, cv2.CV_64F, 0, 1, ksize=3)

        J_I = np.tile(I_dx.reshape(H, W, 1), (1, 1, 6)) * J[:, :, 0, :]\
            + np.tile(I_dy.reshape(H, W, 1), (1, 1, 6)) * J[:, :, 1, :]

        # forward warped image through the network
        net.blobs['data'].data[0] = I
        net.blobs['label'].data[0] = label
        net.forward()
        energy_vec[k] = 100 * net.blobs['prob'].data[0][int(label)].copy()

        if debug:
            fig, axs = plt.subplots(nrows=1, ncols=2, squeeze=False)
            axs[0, 0].imshow(im[0], cmap='gray')
            axs[0, 0].axis('off')
            axs[0, 1].imshow(I, cmap='gray')
            axs[0, 1].axis('off')
            plt.savefig('%s/debug/debug_image_%d_it_%d.png' % (path2save, imID, k))
            plt.close('all')

            fig, axs = plt.subplots(nrows=1, ncols=2, squeeze=False, figsize=(10,10))
            axs[0,0].imshow(np.uint8(I_dx), cmap='gray')
            axs[0,1].imshow(np.uint8(I_dy), cmap='gray')
            fig.savefig('%s/debug/gradient_image_%d_it_%d.png' % (path2save, imID, k))
            plt.close('all')

            print 'it %d , true class probability: ' % k, 100 * net.blobs['prob'].data[0][int(label)]

        if k == K:
            ind = np.argmin(energy_vec)
            theta_final = theta[:, ind]
            I_final = I_vec[ind]
            if debug:
                print 'FINAL score:', energy_vec[ind]
            break  # reache max number of iterations

        net.backward(start='loss')
        im_grad = net.blobs['data'].diff[0].copy()

        # solve LP:
        A = np.dot(im_grad.reshape(1, H*W), J_I.reshape(H*W, 6))
        A = A.reshape(1, 6)

        try:
            model = cplex.Cplex()
            # model.set_problem_type(model.problem_type.LP)
            model.objective.set_sense(model.objective.sense.maximize)
            A = A.reshape(6)
            A = np.concatenate((A, np.zeros(6)), axis=0)
            model.variables.add(obj=A,  types = [model.variables.type.semi_continuous] * 12,
                                names=['theta1', 'theta2', 'theta3', 'theta4', 'theta5', 'theta6',
                                       't1', 't2', 't3', 't4', 't5', 't6'])
            # t1+t2+t3+t4+t5+t6<=(L/K)
            model.linear_constraints.add(lin_expr= [cplex.SparsePair(ind=[6,7,8,9,10,11], val=[1., 1. , 1., 1., 1., 1.])],
                                         senses=["L"],
                                         rhs=[L/float(K)],
                                         names=["l1"])
            # theta_i-t_i <= 0   <=>    theta_i <= t_i
            model.linear_constraints.add(lin_expr= [cplex.SparsePair(ind=[0, 6], val=[1., -1]),
                                                    cplex.SparsePair(ind=[1, 7], val=[1., -1]),
                                                    cplex.SparsePair(ind=[2, 8], val=[1., -1]),
                                                    cplex.SparsePair(ind=[3, 9], val=[1., -1]),
                                                    cplex.SparsePair(ind=[4, 10], val=[1., -1]),
                                                    cplex.SparsePair(ind=[5, 11], val=[1., -1])],
                                                    senses=["L", "L", "L", "L", "L", "L"],
                                                    rhs=np.zeros(6),
                                                    names=["c1", "c2", "c3", "c4", "c5", "c6"])
            # theta_i+t_i >= 0   <=>    theta_i >= -t_i
            model.linear_constraints.add(lin_expr= [cplex.SparsePair(ind=[0, 6], val=[1., 1]),
                                                    cplex.SparsePair(ind=[1, 7], val=[1., 1]),
                                                    cplex.SparsePair(ind=[2, 8], val=[1., 1]),
                                                    cplex.SparsePair(ind=[3, 9], val=[1., 1]),
                                                    cplex.SparsePair(ind=[4, 10], val=[1., 1]),
                                                    cplex.SparsePair(ind=[5, 11], val=[1., 1])],
                                                    senses=["G", "G", "G", "G", "G", "G"],
                                                    rhs=np.zeros(6),
                                                    names=["c7", "c8", "c9", "c10", "c11", "c12"])
            # t_i >=0
            model.linear_constraints.add(lin_expr= [cplex.SparsePair(ind=[6], val=[1.]),
                                                    cplex.SparsePair(ind=[7], val=[1.]),
                                                    cplex.SparsePair(ind=[8], val=[1.]),
                                                    cplex.SparsePair(ind=[9], val=[1.]),
                                                    cplex.SparsePair(ind=[10], val=[1.]),
                                                    cplex.SparsePair(ind=[11], val=[1.])],
                                                    senses=["G", "G", "G", "G", "G", "G"],
                                                    rhs=np.zeros(6),
                                                    names=["c13", "c14", "c15", "c16", "c17", "c18"])

            model.write(os.path.join(path2save, 'adaptiveaugm_2.lp'))
            model.set_log_stream(None)
            model.set_error_stream(None)
            model.set_warning_stream(None)
            model.set_results_stream(None)

            model.solve()
        except CplexError as exc:
            print(exc)
            raise

        # update theta
        delta_theta = np.asarray(model.solution.get_values([0, 1, 2, 3, 4, 5])).\
                                                astype(np.float32).reshape(6,1)
        # assert np.linalg.norm(delta_theta, ord=1) <= L/float(K),\
            # 'Error: parameter vector does not fulfill the condition  |delta_theta|<=L/K'

        theta[:,k+1] = theta[:, k] + delta_theta[:,0]
        # assert np.linalg.norm(theta, ord=1) <= L,\
        #         'Error: parameter vector does not fulfill the condition  |theta|<=L'

        if debug:
            print 'it %d, delta_theta:' % k,  delta_theta.reshape((-1)),
            print 'theta:',  theta[:, k+1].reshape((-1))

        k += 1

    # theta_final = theta
    # I_final = I

    # return theta
    if debug:
        print '============================================='

    return theta_final, I_final

# -------------------------------------------------------------------------------------------------------------------
def adaptiveaugm(net, data, label, p=0.3, K=5):
    N = data.shape[0]
    data_augm = np.zeros(data.shape, dtype=np.float32)
    transf_prob = np.random.rand(N)
    count_augmented = 0
    for i in xrange(N):
        im = data[i]

        if transf_prob[i] < p:
            count_augmented += 1
            theta, im_warp = optimize_param(net, im, i, label[i], K=K, debug=False)
            data_augm[i, 0] = im_warp.astype(np.float32)
        else:
            data_augm[i, 0] = im[0].astype(np.float32)

        # fig, axs = plt.subplots(nrows=1, ncols=2, squeeze=False)
        # axs[0, 0].imshow(im[0], cmap='gray')
        # axs[0, 0].axis('off')
        # axs[0, 1].imshow(data_augm[i, 0], cmap='gray')
        # axs[0, 1].axis('off')
        # plt.savefig('%s/debug/debug_image_%d.png' % (path2save, i))
        # plt.close('all')

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
    if start_snapshot > 0:
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

        tstart = time.time()
        # create a new batch
        data = np.zeros(solver_C.net.blobs['data'].data.shape, dtype=np.float32)
        data[:, :, topleft[0]:topleft[0]+image_size[1], topleft[1]:topleft[1]+image_size[2]] = \
                        np.expand_dims(trainset[data_pointer:data_pointer+batch_size], axis=1)
        labels = trainlabels[data_pointer:data_pointer+batch_size]

        # apply random augmentation of the data
        data_adaptiveaugm, count_augmented = adaptiveaugm(solver_C.net, data, labels, K=K)

        # feead the batch to the classifier
        solver_C.net.blobs['data'].data[...] = data_adaptiveaugm
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
              (epoch_it, it, count_augmented, data_adaptiveaugm.shape[0],
              loss[it-start_snapshot, 0], testloss[it-start_snapshot, 0], tstop - tstart)

        if (it % snapshot_iter == 0 and it > start_snapshot):
            print ' SAVE weights'
            solver_C.net.save(os.path.join(path2save, 'classifier', 'classifier_%d.caffemodel' % it))

        data_pointer += batch_size
        del data_adaptiveaugm

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
