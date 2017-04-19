
import numpy as np
import cv2
import cupy
from skimage import transform
from skimage import exposure
from skimage import util


# -------------------------------------------------------------------------------------------------------------------
def data_preprocess(dataset, datamean=None):
    dataset = dataset.astype('float32')
    if datamean is None:
        datamean = np.expand_dims(np.mean(dataset, axis=0), axis=0).astype('float32')
    dataset = dataset - np.tile(datamean, (dataset.shape[0], 1, 1))
    dataset /= 255.
    return dataset

# -------------------------------------------------------------------------------------------------------------------
def getRotationMatrix2D(angle, scale, transl, shape):
    tx = transl[0]
    ty = transl[1]
    T = np.asarray(
        [[scale * np.cos(angle * np.pi / 180), -scale * np.sin(angle * np.pi / 180), 2 * tx / (shape[1] - 1)],
         [scale * np.sin(angle * np.pi / 180), scale * np.cos(angle * np.pi / 180), 2 * ty / (shape[0] - 1)]])
    return T

# ------------------------------------------------------------------------------------------------
def constr_theta11_theta22(dist, param_trafo):
    D = np.reshape(np.arange(param_trafo['smin'], param_trafo['smax']+0.01, 0.1), (-1, 1)) * \
        np.cos(np.arange(param_trafo['rotmin'], param_trafo['rotmax']+1/180.*np.pi, 1/180.*np.pi))

    if dist == 'min':
        return D.min()

    if dist == 'max':
        return D.max()

# -------------------------------------------------------------------------------------------------------------------
def constr_theta12_theta_21(dist, param_trafo):
    D = np.reshape(np.arange(param_trafo['smin'], param_trafo['smax']+0.01, 0.1), (-1, 1)) * \
        np.sin(np.arange(param_trafo['rotmin'], param_trafo['rotmax']+1/180.*np.pi, 1/180.*np.pi))
    if dist == 'min':
        return D.min()

    if dist == 'max':
        return D.max()

# -------------------------------------------------------------------------------------------------------------------
# def constr_theta13(dist, w, param_trafo):
def constr_theta13(dist, x, y, param_trafo):
    # alpha = np.reshape(np.arange(param_trafo['smin'], param_trafo['smax']+0.01, 0.1), (-1, 1)) * \
    #         np.cos(np.arange(param_trafo['rotmin'], param_trafo['rotmax']+1/180.*np.pi, 1/180.*np.pi))
    # beta = np.reshape(np.arange(param_trafo['smin'], param_trafo['smax']+0.01, 0.1), (-1, 1)) * \
    #        np.sin(np.arange(param_trafo['rotmin'], param_trafo['rotmax']+1/180.*np.pi, 1/180.*np.pi))
    # D = (1 - alpha) * x + beta * y
    # if dist == 'min':
    #     return D.min() - param_trafo['tmax']
    #
    # if dist == 'max':
    #     return D.max() - param_trafo['tmin']

    w = x
    if dist == 'min':
        return 2*param_trafo['tmin'] / (w - 1)
    if dist == 'max':
        return 2*param_trafo['tmax'] / (w - 1)

    # if dist == 'min':
    #     return param_trafo['tmin']
    # if dist == 'max':
    #     return param_trafo['tmax']

# -------------------------------------------------------------------------------------------------------------------
# def constr_theta23(dist, h, param_trafo):
def constr_theta23(dist, x, y, param_trafo):
    # alpha = np.reshape(np.arange(param_trafo['smin'], param_trafo['smax']+0.01, 0.1), (-1, 1)) * \
    #         np.cos(np.arange(param_trafo['rotmin'], param_trafo['rotmax']+1/180.*np.pi, 1/180.*np.pi))
    # beta = np.reshape(np.arange(param_trafo['smin'], param_trafo['smax']+0.01, 0.1), (-1, 1)) * \
    #        np.sin(np.arange(param_trafo['rotmin'], param_trafo['rotmax']+1/180.*np.pi, 1/180.*np.pi))
    # D = - beta * x + (1 - alpha) * y
    # if dist == 'min':
    #     return D.min() - param_trafo['tmax']
    #
    # if dist == 'max':
    #     return D.max() - param_trafo['tmin']

    h = y
    if dist == 'min':
        return 2*param_trafo['tmin'] / (h - 1)
    if dist == 'max':
        return 2*param_trafo['tmax'] / (h - 1)

    # if dist == 'min':
    #     return param_trafo['tmin']
    # if dist == 'max':
    #     return param_trafo['tmax']
# -------------------------------------------------------------------------------------------------------------------
def get_trueclass_probability(net, mean, im, label):
    net.blobs['data'].data[0] = im - mean
    net.blobs['label'].data[0] = label
    net.forward()
    return 100 * net.blobs['prob'].data[0][int(label)]

# -------------------------------------------------------------------------------------------------------------------
def warpAffine_gpu(im, T, out_size, gpuID=0):

    H = im.shape[0]
    W = im.shape[1]
    N = np.prod(out_size)

    x_t, y_t = np.meshgrid(np.linspace(-1, 1, out_size[1]),
                           np.linspace(-1, 1, out_size[0]))
    with cupy.cuda.Device(gpuID):

        im_gpu = cupy.array(im)

        x_t = cupy.array(x_t, cupy.float32)
        y_t = cupy.array(y_t, cupy.float32)

        source_grid = cupy.vstack([x_t.flatten(), y_t.flatten(), cupy.ones(np.prod(x_t.shape))])
        T_gpu = cupy.array(T, cupy.float32)

        target_grid = cupy.dot(T_gpu, source_grid)
        x_s = cupy.reshape(target_grid[0, :], (-1))
        y_s = cupy.reshape(target_grid[1, :], (-1))

        del target_grid, source_grid

        x_s = cupy.array((x_s + 1.0) * (W - 1) / 2.0, cupy.float32)
        y_s = cupy.array((y_s + 1.0) * (H - 1) / 2.0, cupy.float32)

        x0 = cupy.array(cupy.floor(x_s), cupy.int32)
        x1 = x0 + 1
        y0 = cupy.array(cupy.floor(y_s), cupy.int32)
        y1 = y0 + 1

        x0 = cupy.clip(x0, a_min=0, a_max=W-1)
        x1 = cupy.clip(x1, a_min=0, a_max=W-1)
        y0 = cupy.clip(y0, a_min=0, a_max=H-1)
        y1 = cupy.clip(y1, a_min=0, a_max=H-1)

        ind_x0y0 = y0 * W + x0
        ind_x0y1 = y1 * W + x0
        ind_x1y0 = y0 * W + x1
        ind_x1y1 = y1 * W + x1

        x0_f = cupy.array(x0, cupy.float32)
        x1_f = cupy.array(x1, cupy.float32)
        y0_f = cupy.array(y0, cupy.float32)
        y1_f = cupy.array(y1, cupy.float32)

        im_out_gpu = cupy.zeros((N, 1), cupy.float32)

        val_orig = cupy.reshape(im_gpu[:, :], (-1))
        im_out_gpu = val_orig[ind_x0y0] * (x1_f - x_s) * (y1_f - y_s)\
                    + val_orig[ind_x0y1] * (x1_f - x_s) * (y_s - y0_f)\
                    + val_orig[ind_x1y0] * (x_s - x0_f) * (y1_f - y_s)\
                    + val_orig[ind_x1y1] * (x_s - x0_f) * (y_s - y0_f)

        del ind_x0y0, ind_x0y1
        im_out_gpu = im_out_gpu.reshape(out_size[0], out_size[1])
        im_out = im_out_gpu.get()

    return im_out
