# This defines a simple autoencoder model with one hidden layer for computing embedding on a graph.
# It experiments different activations, loss functions and implementations

import mxnet as mx
import numpy as np
import scipy as sp
import sys
import logging
import time
import math
import matplotlib.pyplot as plt

def get_act(act):
    if (act == 'sigmoid'):
        return sp.special.expit
    elif (act == 'tanh'):
        return np.tanh
    elif (act == 'relu'):
        return lambda x: np.maximum(x, 0)
    else:
        return None
    
weight_names = ['fc1_weight', 'fc2_weight']
bias_names = ['fc1_bias', 'fc2_bias']

def SparseSyms(dims):
    data = mx.symbol.Variable('data', stype='csr')
    y = mx.symbol.Variable('label')
    fc1_weight = mx.symbol.Variable(weight_names[0], shape=(dims[0], dims[1]), stype='row_sparse')
    fc1_bias = mx.symbol.Variable(bias_names[0], shape=(dims[1]))
    fc2_weight = mx.symbol.Variable(weight_names[1], shape=(dims[2], dims[1]))
    fc2_bias = mx.symbol.Variable(bias_names[1], shape=(dims[2]))
    return (data, y, fc1_weight, fc1_bias, fc2_weight, fc2_bias)

def DenseSyms(dims):
    data = mx.symbol.Variable('data')
    y = mx.symbol.Variable('label')
    fc1_weight = mx.symbol.Variable(weight_names[0], shape=(dims[1], dims[0]))
    fc1_bias = mx.symbol.Variable(bias_names[0], shape=(dims[1]))
    fc2_weight = mx.symbol.Variable(weight_names[1], shape=(dims[2], dims[1]))
    fc2_bias = mx.symbol.Variable(bias_names[1], shape=(dims[2]))
    return (data, y, fc1_weight, fc1_bias, fc2_weight, fc2_bias)

def SparseFC(data, weight, bias, num_hidden):
    dot = mx.symbol.sparse.dot(data, weight)
    return mx.symbol.broadcast_add(dot, bias)

def DenseFC(data, weight, bias, num_hidden):
    return mx.symbol.FullyConnected(data=data, weight=weight, bias=bias,
                                    num_hidden=num_hidden)

def DenseNumpyFC(data, weight, bias):
    return np.dot(data, weight.T) + bias

def SparseNumpyFC(data, weight, bias):
    return np.dot(data, weight) + bias

def log_loss(x, y):
    return -(y * np.log(x) + (1 - y) * np.log(1 - x))

def logistic_np(x, y):
    return np.sum(log_loss(sp.special.expit(x), y))

def logistic_mx(x, y):
    x = 1/(1+mx.symbol.exp(-x))
    return mx.symbol.sum(-(y * mx.symbol.log(x) + (1 - y) * mx.symbol.log(1 - x)))

def get_loss(name):
    if (name == "L2"):
        l2_numpy = lambda x, y: np.sum(np.square(x - y))
        l2_mx = lambda x, y: mx.symbol.LinearRegressionOutput(data=x, label=y)
        return l2_numpy, l2_mx
    elif (name == "logistic"):
        sm_numpy = logistic_np
        sm_mx = logistic_mx
        return sm_numpy, sm_mx
    else:
        return None

def plot_errors(x, y):
    plt.plot(x, y)
    plt.ylabel('error')
    plt.xlabel('epoch')
    plt.show()

class AutoEncoderModel:
    def __init__(self, data, y, num_dims, internal_act=None, learning_rate=0.005,
                 batch_size=50, loss_name="L2", proj=None, use_sparse=False):
        dims = [data.shape[1], num_dims, y.shape[1]]
        if (use_sparse):
            print("Use sparse operators")
            CreateSyms = SparseSyms
            FC = SparseFC
            NumpyFC = SparseNumpyFC
        else:
            CreateSyms = DenseSyms
            FC = DenseFC
            NumpyFC = DenseNumpyFC
            
        syms = CreateSyms(dims)
        self.data = syms[0]
        self.y = syms[1];
        self.fc1_weight = syms[2]
        self.fc1_bias = syms[3]
        self.fc2_weight = syms[4]
        self.fc2_bias = syms[5]
        x = FC(data=self.data, weight=self.fc1_weight, bias=self.fc1_bias, num_hidden=dims[1])
        if (internal_act is not None):
            x = mx.symbol.Activation(data=x, act_type=internal_act)
            print("Internal activation: " + internal_act)
        self.layer1 = x
        x = DenseFC(data=x, weight=self.fc2_weight, bias=self.fc2_bias, num_hidden=dims[2])
        self.layer2 = x
        print("loss func: " + loss_name)
        np_loss, mx_loss = get_loss(loss_name)
        # TODO How about using L1/L2 regularization.
        self.loss = mx_loss(x, self.y)
        self.model = mx.mod.Module(symbol=self.loss, data_names=['data'], label_names = ['label'])
        self.init_data(data, y, batch_size, learning_rate)
        
        def cal_model_numpy(params):
            fc1_weight = params.get(weight_names[0]).asnumpy()
            fc1_bias = params.get(bias_names[0]).asnumpy()
            fc2_weight = params.get(weight_names[1]).asnumpy()
            fc2_bias = params.get(bias_names[1]).asnumpy()

            np_data = data.asnumpy()
            hidden = NumpyFC(np_data, fc1_weight, fc1_bias)
            act_func = get_act(internal_act)
            if (act_func is not None):
                hidden = act_func(hidden)
            output = DenseNumpyFC(hidden, fc2_weight, fc2_bias)
            if (proj is not None):
                output = np.dot(output, proj)
            return hidden, output, np_loss(output, np_data)

        self.numpy_cal = cal_model_numpy
        
    def init_data(self, data, y, batch_size=50, learning_rate=0.005):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        data_iter = mx.io.NDArrayIter(data={'data':data}, label={'label':y},
                batch_size=batch_size, shuffle=True,
                last_batch_handle='discard')
        print("Learning rate: " + str(learning_rate))
        print("batch size: " + str(batch_size))
        # allocate memory given the input data and label shapes
        self.model.bind(data_shapes=data_iter.provide_data, label_shapes=data_iter.provide_label)
        # initialize parameters by uniform random numbers
        self.model.init_params(initializer=mx.init.Uniform(scale=.1))
        # use SGD with learning rate 0.1 to train
        self.model.init_optimizer(optimizer='sgd',
                                  optimizer_params={'learning_rate': learning_rate,
                                                    'momentum': 0.9})

    def fit_int(self, data_iter, params=None, learning_rate=0.005, reinit_opt=True):
        if (params is not None):
            self.model.set_params(arg_params=params, aux_params=None, force_init=True)
        if (reinit_opt):
            print("reinit optimizer. New learning rate: " + str(learning_rate))
            self.model.init_optimizer(optimizer='sgd',
                                      optimizer_params={'learning_rate': learning_rate,
                                                        'momentum': 0.9}, force_init=True)
        feval = lambda x, y : ((x - y)*(x-y)).sum()
        self.metric = mx.metric.create(feval)
        data_iter.reset()
        self.metric.reset()
        for batch in data_iter:
            self.model.forward(batch, is_train=True)       # compute predictions
            self.model.update_metric(self.metric, batch.label)  # accumulate prediction accuracy
            self.model.backward()                          # compute gradients
            self.model.update()                            # update parameters

    def train(self, data, y, num_epoc, params = None, debug=False, return_err=False, plot=False):
        print("internal #epochs: " + str(num_epoc))
        prev_val = None
        reinit_opt = True
        plot_xs = []
        plot_yx = []
        data_iter = mx.io.NDArrayIter(data={'data':data}, label={'label':y},
                batch_size=self.batch_size, shuffle=True,
                last_batch_handle='discard')
        for i in range(num_epoc - 1):
            self.fit_int(data_iter, params, self.learning_rate, reinit_opt=reinit_opt)
            params = None
            reinit_opt = False

            val = self.metric.get()[1]
            plot_xs.append(i + 1)
            plot_yx.append(val)
            if (debug):
                print("epoc " + str(i + 1) + ": " + str(val))
                sys.stdout.flush()
            if (prev_val is not None and prev_val < val):
                self.learning_rate = self.learning_rate / 2
                reinit_opt = True
            prev_val = val
        if (plot):
            plot_errors(plot_xs, plot_yx)
        params = self.model.get_params()[0]
        if (return_err):
            return params, plot_xs, plot_yx
        else:
            return params