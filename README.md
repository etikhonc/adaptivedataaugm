## Adaptive data augmentation for image classification
This code is a replicattion of the method suggested in

"Adaptive data augmentation for image classification"
by A. Fawzi, H.Samulowitz, D.Turaga, P.Frossard, EPFL&IBM Watson Research, ICIP 2016

## Dataset
We use MNIST dataset. We randomly sampled 5000 image from the training set which
we utilize as a new training set. Images are rescaled to 16x16 and put
in the middle of the input batch of a network.
For testing we use the whole MNIST test set.

## Baseline
We utilize LeNet-1 architecture: conv-pool-relu-fc. We use batch size 100.

## Definition of the transformation
We consider a class of affine transformations. We represent one transformation
with 4 parameters theta\in \mathbb{R}^{4}:
theta1 -theta2 theta3
theta2 theta1  theta4

The set of feasible transformations is defined through the boundaries of allowed
rotation, scale and translation values.

## Randaugm(S)
Train baseline network from scratch using random augmentation:
randomly pick a feasible rotation angle, scale and translation vector
and warp image using the selected values.

## Randaugm(F)
Finetune baseline network after 40 epoch of training. The selection of
feasible transformations is same as in Randaugm(S).

## Randaugm(F)2
Finetune baseline network after 40 epoch of training.

## Adaptiveaugm
Finetune baseline network after 40 epoch of training. Use adaptive augmentation
algorithm from the paper to augment 30% of images in each batch. However
we do not constrain the norm of the parameter vector to be small than L.
Instead we constrain a feasible range for the rotation, translation and
scale components of an affine transformation.


## Adaptiveaugm2
Finetune baseline network after 40 epoch of training. Use adaptive augmentation
algorithm from the paper to augment 30% of images in each batch. 

Here similar to Randaugm(F)2 we consider \theta\in\mathbb{R}^6 (as in the original paper) and constraing the norm of
the parameter vector theta be smaller than 0.25, instead of constraining feasible range of rotation, scale
and translation vectors.


## Results

    |   method      | test_err | top1-acc | top5-acc |
    | ------------- | -------- | -------- | -------- |
    | baseline      |   3.06%  |  96.94%  | 99.98%   |
    | ------------- | -------- | -------- | -------- |
    | randaugm      |   2.90%  |  97.10%  | 99.96%   | warpAffine_gpu
    | randaugm      |   2.62%  |  97.38%  | 99.98%   | cv2.warpAffine
    | ------------- | -------- | -------- | -------- |
    | randaugm(F)   |   2.81%  |  97.19%  | 99.98%   | warpAffine_gpu
    | randaugm(F)   |   2.49%  |  97.51%  | 99.98%   | cv2.warpAffine
    | ------------- | -------- | -------- | -------- |
    | randaugm(F)2  |   2.78%  |  97.22%  | 99.96%   | warpAffine_gpu
    | randaugm(F)2  |   2.94%  |  97.06%  | 99.98%   | cv2.warpAffine
    | ------------- | -------- | -------- | -------- |
    | adaptiveaugm  |   3.08%  |  96.92%  | 99.96%   | cv2.warpAffine
    | adaptiveaugm2 |   3.60%  |  96.40%  | 99.92%   | cv2.warpAffine
    | ------------- | -------- | -------- | -------- |
