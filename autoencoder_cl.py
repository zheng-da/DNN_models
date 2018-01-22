# This file provides code for curriculum learning.
# Here we start with a small model and extend the trained small model to a larger model.

import mxnet as mx
import numpy as np
import scipy as sp
import sys
import logging
import time
import math
import gc
import copy
from autoencoder import AutoEncoderModel
from autoencoder import weight_names, bias_names
from autoencoder import AutoEncoderModel
from autoencoder import weight_names, bias_names

def init_params_svd(data, num_hidden):
    num_inputs = data.shape[1]
    U, s, Vh = sp.sparse.linalg.svds(data, k=num_hidden)
    
    new_params = {}
    new_params.update({weight_names[0]: mx.ndarray.array(Vh)})
    new_params.update({bias_names[0]: mx.ndarray.random_uniform(shape=(num_hidden,))})
    new_params.update({weight_names[1]: mx.ndarray.array(Vh.T)})
    new_params.update({bias_names[1]: mx.ndarray.random_uniform(shape=(num_inputs,))})
    return new_params

# Extend the weight matrices in the encoder of the smaller autoencoder to
# the shape required by the larger autoencoder.
def extend_params_encode(weight, bias, idx, num_inputs, num_outputs, init):
    if (isinstance(init, np.ndarray)):
        out_weight = init
    else:
        # We initialize the weights in the same way as MXNet
        out_weight = np.random.uniform(low=-0.1, high=0.1, size=(num_outputs, num_inputs))
    out_bias = np.zeros(num_outputs)
    out_weight[0:weight.shape[0], idx] = weight.asnumpy()
    out_bias[0:bias.shape[0]] = bias.asnumpy()
    return mx.nd.array(out_weight), mx.nd.array(out_bias)

# Extend the weight matrices in the decoder of the smaller autoencoder to
# the shape required by the larger autoencoder.
def extend_params_decode(weight, bias, idx, num_inputs, num_outputs, init):
    if (isinstance(init, np.ndarray)):
        out_weight = init
    else:
        # We initialize the weights in the same way as MXNet
        out_weight = np.random.uniform(low=-0.1, high=0.1, size=(num_outputs, num_inputs))
    out_bias = np.zeros(num_outputs)
    out_weight[idx, 0:weight.shape[1]] = weight.asnumpy()
    out_bias[idx] = bias.asnumpy()
    return mx.nd.array(out_weight), mx.nd.array(out_bias)

# This function extends the parameter matrices in the small autoencoder
# to an autoencoder with the specified number of input nodes and hidden nodes.
def extend_params(params, new_data, new_hidden, new_outputs, rand_init=False):
    old_inputs = params.get(weight_names[0]).shape[1]
    old_hidden = params.get(bias_names[0]).shape[0]
    new_inputs = new_data.shape[1]
    max_cols, max_idx = get_densest_idx(new_data, old_inputs)
    
    num_inputs = new_data.shape[1]
    if (rand_init):
        init = True
    else:
        U, s, Vh = sp.sparse.linalg.svds(new_data, k=new_hidden)
        init = Vh
    weight, bias = extend_params_encode(params.get(weight_names[0]),
                                        params.get(bias_names[0]), max_idx,
                                        new_inputs, new_hidden, init)
    new_params = {}
    new_params.update({weight_names[0]: weight})
    new_params.update({bias_names[0]: bias})
    if (rand_init):
        init = True
    else:
        init = Vh.T
    weight, bias = extend_params_decode(params.get(weight_names[1]),
                                        params.get(bias_names[1]), max_idx,
                                        new_hidden, new_outputs, init)
    new_params.update({weight_names[1]: weight})
    new_params.update({bias_names[1]: bias})
    return new_params

def get_densest_idx(spm, num):
    colsum = np.ravel(spm.sum(axis=0))
    max_cols = np.sort(np.ravel(colsum), axis=None)[len(colsum) - num]
    return max_cols, colsum >= max_cols

def get_densest(spm, num):
    max_cols, idx = get_densest_idx(spm, num)
    sp_data = spm[:,idx]
    return sp_data

def get_densest2(spm, num1, num2):
    colsum = np.ravel(spm.sum(axis=1))
    sorted_colsum = np.sort(np.ravel(colsum), axis=None)
    max_cols1 = sorted_colsum[len(colsum) - num1]
    max_cols2 = sorted_colsum[len(colsum) - (num2 + num1)]
    sp_data1 = spm[:,colsum >= max_cols1]
    sp_data2 = spm[:,np.logical_and(colsum >= max_cols2, colsum < max_cols1)]
    return sp_data1, sp_data2

def proj_back_params(params, Vh):
    weight = params.get(weight_names[1])
    weight = mx.ndarray.dot(mx.ndarray.array(Vh.T), weight)
    bias = params.get(bias_names[1])
    params_cpy = copy.copy(params)
    params_cpy.update({weight_names[1]: weight})
    params_cpy.update({bias_names[1]: mx.ndarray.dot(mx.ndarray.array(Vh.T), bias)})
    return params_cpy

def proj_params(params, Vh):
    weight = params.get(weight_names[1])
    bias = params.get(bias_names[1])
    params_cpy = copy.copy(params)
    params_cpy.update({weight_names[1]: mx.ndarray.dot(mx.ndarray.array(Vh), weight)})
    params_cpy.update({bias_names[1]: mx.ndarray.dot(mx.ndarray.array(Vh), bias)})
    return params_cpy

def train_cl(spm, target_ndims, target_red_ndims, num_epoc, init_ndims=40,
             init_red_ndims=20, num_incs = 10, iact='relu', learning_rate=0.8,
             lr_inc=2, approx_loss=False, use_sparse=False):
    learning_rate = float(learning_rate)
    spm = get_densest(spm, target_ndims)
    target_ndims = spm.shape[1]
    orig_model = AutoEncoderModel(mx.ndarray.sparse.csr_matrix(spm), spm,
                                  num_dims=target_red_ndims, internal_act=iact,
                                  learning_rate=learning_rate, batch_size=2000,
                                  use_sparse=use_sparse)

    ndims = init_ndims
    red_ndims = init_red_ndims
    ndims_inc_exp = math.pow((float(target_ndims)/init_ndims), 1/float(num_incs))
    red_ndims_inc = (float(target_red_ndims) - init_red_ndims) / num_incs
    print("#dims increase by a factor of " + str(ndims_inc_exp))
    print("#red dims increase by " + str(red_ndims_inc))
    
    # This is the first run to train the small model.
    print("#dims: " + str(ndims))
    print("reduce #dims to " + str(red_ndims))
    sp_data = get_densest(spm, ndims)
    U, s, Vh = sp.sparse.linalg.svds(sp_data, k=red_ndims)
    res = np.dot(sp_data.dot(Vh.T), Vh)
    print("svd error: " + str(np.sum(np.square(res - sp_data))))
    data = mx.ndarray.sparse.csr_matrix(sp_data)
    y = mx.ndarray.dot(data, mx.ndarray.array(Vh.T))
    y = data
    model = AutoEncoderModel(data, y, num_dims=red_ndims, internal_act=iact,
                             learning_rate=learning_rate, batch_size=2000, use_sparse=use_sparse)
    params, _, errors = model.train(data, y, num_epoc=num_epoc, return_err=True)

    prev_Vh = None
    for i in range(num_incs):
        ndims = int(ndims * ndims_inc_exp)
        red_ndims = int(red_ndims + red_ndims_inc)
        print("#dims: " + str(ndims))
        print("reduce #dims to " + str(red_ndims))
        sp_data = get_densest(spm, ndims)
        U, s, Vh = sp.sparse.linalg.svds(sp_data, k=red_ndims)
        res = np.dot(sp_data.dot(Vh.T), Vh)
        print("svd error: " + str(np.sum(np.square(res - sp_data))))
        print("")

        start = time.time()
        data = mx.ndarray.sparse.csr_matrix(sp_data)
        Vh = None
        if (approx_loss and red_ndims * 1.5 < sp_data.shape[1]):
            _, _, Vh = sp.sparse.linalg.svds(sp_data, k=int(red_ndims * 1.5))
            y = mx.ndarray.dot(data, mx.ndarray.array(Vh.T))
            print("Approximate loss with " + str(y.shape[1]) + " dims")
        else:
            y = data
        
        # We use the previously trained model to initialize the current model.
        print("Train from previous results")
        # If we use sparse operations, we need to flip the weight matrix in the first layer
        # because the following operations (projection and parameter extension)
        # always assume the first dimension of the weight is for the hidden layer
        # and the second dimension is for the input layer.
        # TODO flipping a matrix can be expensive if the weight matrix is
        # very large. I'll fix it later.
        if (use_sparse):
            weight = params.get(weight_names[0])
            params.update({weight_names[0]: weight.T})
        if (prev_Vh is not None):
            params = proj_back_params(params, prev_Vh)
        params_init = extend_params(params, sp_data, red_ndims, data.shape[1], rand_init=True)
        if (Vh is not None):
            params_init = proj_params(params_init, Vh)
            prev_Vh = Vh
        if (use_sparse):
            weight = params_init.get(weight_names[0])
            params_init.update({weight_names[0]: weight.T})
        
        model = AutoEncoderModel(data, y, num_dims=red_ndims, internal_act=iact, learning_rate=learning_rate,
                                 batch_size=2000, proj=Vh, use_sparse=use_sparse)
        params, _, errors = model.train(data, y, num_epoc=num_epoc, return_err=True,
                                        params=params_init)
        # If the learning rate wasn't reduced during the training, we can increase
        # the learning rate for the next larger model.
        if (learning_rate == model.learning_rate):
            learning_rate = learning_rate * lr_inc
        print("It takes " + str(time.time() - start) + " seconds")
        
        _, _, tot_loss = model.numpy_cal(params)
        print("The error: " + str(tot_loss))
        orig_params = params
        if (Vh is not None):
            orig_params = proj_back_params(params, Vh)
        orig_params = extend_params(orig_params, spm, target_red_ndims, target_ndims, rand_init=True)
        _, _, orig_loss = orig_model.numpy_cal(orig_params)
        print("The original error: " + str(orig_loss))
        print("")
        gc.collect()

    return model