#!/usr/bin/env python
#coding=utf-8

import numpy as np
from typing import List
import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    InputLayer,
    Flatten,
    Activation,
    Dense
)
from tensorflow.keras.layers import (
    Conv2D, 
    Conv2DTranspose,
    MaxPooling2D, 
    UpSampling2D
)
from tensorflow.keras.models import Model, Sequential
import tensorflow.keras.backend as K


class VisLayer():
    ''' Abstract base class for deconv layers'''
    up_func = lambda arg: None
    down_func = lambda arg: None
    name: str = ''

    def up(self, data, learning_phase = 0):
        self.up_data = self.up_func([data, learning_phase])
        return self.up_data

    def down(self, data, learning_phase = 0):
        self.down_data= self.down_func([data, learning_phase])
        return self.down_data


class VisConv2D(VisLayer):
    
    def __init__(self, layer):

        self.layer = layer
        self.name = layer.name

        W, b = layer.get_weights()
        i = Input(shape = layer.input_shape[1:])
        u = Conv2D(
            filters = layer.filters, 
            kernel_size = layer.kernel_size, 
            strides = layer.strides,
            dilation_rate = layer.dilation_rate,
            padding = layer.padding,
            kernel_initializer = tf.constant_initializer(W),
            bias_initializer = tf.constant_initializer(b),
        )
        self.up_func = K.function([i, K.learning_phase()], u(i))

        W_t = np.moveaxis(W[::-1, ::-1, :, :], 2, 3) 
        b_t = np.zeros(W_t.shape[3])
        o = Input(shape = layer.output_shape[1:])
        d = Conv2D(
            filters = W_t.shape[-1], 
            kernel_size = W_t.shape[:2], 
            strides = layer.strides,
            dilation_rate = layer.dilation_rate,
            padding = layer.padding,
            kernel_initializer=tf.constant_initializer(W_t),
            bias_initializer=tf.constant_initializer(b_t),
        )
        self.down_func = K.function([o, K.learning_phase()], d(o))


class VisDense(VisLayer):
    def __init__(self, layer: Dense):
        self.layer = layer
        self.name = layer.name

        weights = layer.get_weights()
        W = weights[0]
        b = weights[1]
        
        #Set up_func for DDense
        i = Input(shape = layer.input_shape[1:])
        o = Dense(units = layer.output_shape[1])(i)
        o.weights = [W, b]
        self.up_func = K.function([i, K.learning_phase()], o)
        
        #Transpose W and set down_func for DDense
        W = W.transpose()
        self.input_shape = layer.input_shape
        self.output_shape = layer.output_shape
        b = np.zeros(self.input_shape[1])
        flipped_weights = [W, b]
        i = Input(shape = self.output_shape[1:])
        o = Dense(units = self.input_shape[1])(i)
        o.weights = flipped_weights
        self.down_func = K.function([i, K.learning_phase()], o)


class VisMaxPooling2D(VisLayer):
    def __init__(self, layer: MaxPooling2D):
        self.layer = layer
        self.name = layer.name
        self.poolsize = layer.pool_size

        # set up functions
        inp = Input(shape = layer.input_shape[1:])
        pool = MaxPooling2D(
            pool_size = layer.pool_size,
            strides = layer.strides,
            padding = layer.padding,
            data_format = 'channels_last'
        )
        self._maxpool = K.function([inp, K.learning_phase()], pool(inp))

        out = inp = Input(shape = layer.output_shape[1:])
        ups_factor = int(layer.input_shape[1]/layer.output_shape[1])
        ups = UpSampling2D(size=(ups_factor, ups_factor))
        self._upsample = K.function([out, K.learning_phase()], ups(out))
        
        self._switches = np.zeros((1, *layer.input_shape[1:]))
    

    def up(self, data, learning_phase = 0):
        self.up_data = self._maxpool([data, learning_phase])
        self._switches = data == self._upsample([self.up_data, learning_phase])
        return self.up_data
    

    def down(self, data, learning_phase = 0):
        self.down_data = self._upsample([data, learning_phase]) * self._switches
        return self.down_data


class VisActivation(VisLayer):
    def __init__(self, layer: Activation, linear = False):
        self.layer = layer
        self.name = layer.name
        self.linear = linear
        self.activation = layer.activation
        i = K.placeholder(shape = layer.output_shape)

        a = self.activation(i)
        # According to the original paper, 
        # In forward pass and backward pass, do the same activation(relu)
        self.up_func = K.function([i, K.learning_phase()], a)
        self.down_func = K.function([i, K.learning_phase()], a)
    
    
class VisFlatten(VisLayer):
    def __init__(self, layer: Flatten):
        self.layer = layer
        self.name = layer.name
        self.shape = layer.input_shape[1:]
        self.up_func = K.function([layer.input, K.learning_phase()], layer.output)

    # Flatten 2D input into 1D output
    def up(self, data, learning_phase = 0):
        self.up_data = self.up_func([data, learning_phase])
        return self.up_data

    # Reshape 1D input into 2D output
    def down(self, data, learning_phase = 0):
        new_shape = [data.shape[0]] + list(self.shape)
        assert np.prod(self.shape) == np.prod(data.shape[1:])
        self.down_data = np.reshape(data, new_shape)
        return self.down_data


class VisInput(VisLayer):
    def __init__(self, layer: Input):
        self.layer = layer
        self.name = layer.name
    
    # input and output of Inputl layer are the same
    def up(self, data, learning_phase = 0):
        self.up_data = data
        return self.up_data
    
    def down(self, data, learning_phase = 0):
        self.down_data = data
        return self.down_data


class VisModel():
    layers: List[VisLayer] = []

    def __init__(self, model: Model
                     , layer_name = ''):
        self.layer_name = layer_name

        for l in model.layers:
            if   isinstance(l, Conv2D):
                self.layers.append(VisConv2D(l))
                self.layers.append(VisActivation(l))
            elif isinstance(l, MaxPooling2D):
                self.layers.append(VisMaxPooling2D(l))
            elif isinstance(l, Dense):
                self.layers.append(VisDense(l))
                self.layers.append(VisActivation(l))
            elif isinstance(l, Activation):
                self.layers.append(VisActivation(l))
            elif isinstance(l, Flatten):
                self.layers.append(VisFlatten(l))
            elif isinstance(l, InputLayer):
                self.layers.append(VisInput(l))
            else:
                raise ValueError(f'Cannot handle this type of layer: \n{l.get_config()}')
            if l.name == layer_name:
                break

    
    def visualize(self, data
                      , top_n: int
                      , max_only = False
                      , save_img = lambda img, name: None ):
        up_data = self.up(data)
        for rank, feat, d in self.get_top_features(up_data, top_n, max_only):
            down_data = self.down(d)
            save_img(np.array(down_data).squeeze(), rank+1, feat)


    def up(self, data):
        self.layers[0].up(data)
        for j in range(1, len(self.layers)):
            self.layers[j].up(self.layers[j - 1].up_data)

        return self.layers[-1].up_data


    def get_top_features(self, data, n: int, max_only = False):
        ''' Returns iterable yielding tuples of (ranking, feature_index, output_data)'''

        # compute max for each feature
        f_max = np.array(list(map( 
            lambda f: data[...,f].max(),
            range(data.shape[-1]),
        )))

        # get top n features
        i_max_n = np.argpartition(f_max, -n)[-n:]
        f_max_n = f_max[i_max_n]
        i_max_n_desc = i_max_n[np.argsort(f_max_n[::-1])]

        for j, ind in enumerate(i_max_n_desc):
            o = np.zeros_like(data)
            feature_map = data[..., ind]
            if max_only:
                feature_map = feature_map * (feature_map == feature_map.max())
            o[..., ind] = feature_map
            yield (j, ind, o)



    def down(self, data):
        self.layers[-1].down(data)
        for j in reversed(range(len(self.layers[:-1]))):
            self.layers[j].down(self.layers[j+1].down_data)

        return self.layers[0].down_data
        