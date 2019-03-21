#!/usr/bin/env python
#coding=utf-8
###########################################################
# Modified from https://github.com/jalused/Deconvnet-keras/
###########################################################

import argparse
import numpy as np
import sys
from PIL import Image
from typing import (List, Tuple, Callable)
from tensorflow.keras.layers import (
    Input,
    InputLayer,
    Flatten,
    Activation,
    Dense
)
from tensorflow.keras.layers import (
    Convolution2D,
    MaxPooling2D
)
from tensorflow.keras.activations import *
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.applications import vgg16
import tensorflow.keras.backend as K


class DLayer():
    ''' Abstract base class for deconv layers'''
    up_func: Callable = lambda arg: None
    down_func: Callable = lambda arg: None
    name: str = ''

    def up(self, data, learning_phase = 0):
        self.up_data = self.up_func([data, learning_phase])
        return self.up_data

    def down(self, data, learning_phase = 0):
        self.down_data= self.down_func([data, learning_phase])
        return self.down_data


class DConvolution2D(DLayer):
    def __init__(self, layer: Convolution2D):
        self.layer = layer
        self.name = layer.name

        weights = layer.get_weights()
        W = weights[0]
        b = weights[1]

        # Set up_func for DConvolution2D
        up_kernel_size = tuple(W.shape[0:2])
        up_filters = W.shape[3]
        i = Input(shape = layer.input_shape[1:])
        o = Convolution2D(
            filters = up_filters, 
            kernel_size = up_kernel_size, 
            padding = 'same',
        )(i)
        o.weights=[W,b]
        self.up_func = K.function([i, K.learning_phase()], o)

        # Flip W horizontally and vertically, 
        # and set down_func for DConvolution2D
        W = np.transpose(W, (0, 1, 3, 2))
        W = W[::-1, ::-1, :, :]
        down_kernel_size = tuple(W.shape[0:2])
        down_filters = W.shape[3]
        b = np.zeros(down_filters)
        i = Input(shape = layer.output_shape[1:])
        o = Convolution2D(
            filters = down_filters, 
            kernel_size = down_kernel_size, 
            padding = 'same',
        )(i)
        o.weights = [W,b]
        self.down_func = K.function([i, K.learning_phase()], o)
    

class DDense(DLayer):
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
    

class DPooling(DLayer):
    def __init__(self, layer: MaxPooling2D):
        self.layer = layer
        self.name = layer.name
        self.poolsize = layer.pool_size
    
    def up(self, data, learning_phase = 0):
        [self.up_data, self.switch] = self.__max_pooling_with_switch(data, self.poolsize)
        return self.up_data

    def down(self, data, learning_phase = 0):
        self.down_data = self.__max_unpooling_with_switch(data, self.switch)
        return self.down_data

    def __max_pooling_with_switch(self, inp, poolsize):
        '''
        Compute pooling output and switch in forward pass, switch stores 
        location of the maximum value in each poolsize * poolsize block
        # Arguments
            inp: data to be pooled
            poolsize: size of pooling operation
        # Returns
            Pooled result and Switch
        '''
        i = np.moveaxis(inp,3,1)

        switch = np.zeros(i.shape)
        out_shape = list(i.shape)
        row_poolsize = int(poolsize[0])
        col_poolsize = int(poolsize[1])
        out_shape[2] = int(out_shape[2] / poolsize[0])
        out_shape[3] = int(out_shape[3] / poolsize[1])
        pooled = np.zeros(out_shape)
        
        for sample in range(i.shape[0]):
            for dim in range(i.shape[1]):
                for row in range(out_shape[2]):
                    for col in range(out_shape[3]):
                        patch = i[sample, 
                                dim, 
                                row * row_poolsize : (row + 1) * row_poolsize,
                                col * col_poolsize : (col + 1) * col_poolsize]
                        max_value = patch.max()
                        pooled[sample, dim, row, col] = max_value
                        max_col_index = patch.argmax(axis = 1)
                        max_cols = patch.max(axis = 1)
                        max_row = max_cols.argmax()
                        max_col = max_col_index[max_row]
                        switch[sample, 
                                dim, 
                                row * row_poolsize + max_row, 
                                col * col_poolsize + max_col]  = 1
        
        pooled = np.moveaxis(pooled,1,3)
        switch = np.moveaxis(switch,1,3)
        return [pooled, switch]


    # Compute unpooled output using pooled data and switch
    def __max_unpooling_with_switch(self, inp, switch):
        '''
        Compute unpooled output using pooled data and switch
        # Arguments
            inp: data to be pooled
            poolsize: size of pooling operation
            switch: switch storing location of each elements
        # Returns
            Unpooled result
        '''
        i = np.moveaxis(inp,3,1)
        switch = np.moveaxis(switch,3,1)

        tile = np.ones((int(switch.shape[2] / i.shape[2]), 
                        int(switch.shape[3] / i.shape[3])))
        out = np.kron(i, tile)
        unpooled = out * switch

        unpooled = np.moveaxis(unpooled,1,3)
        return unpooled



class DActivation(DLayer):
    def __init__(self, layer: Activation, linear = False):
        self.layer = layer
        self.name = layer.name
        self.linear = linear
        self.activation = layer.activation
        i = K.placeholder(shape = layer.output_shape)

        o = self.activation(i)
        # According to the original paper, 
        # In forward pass and backward pass, do the same activation(relu)
        self.up_func = K.function([i, K.learning_phase()], o)
        self.down_func = K.function([i, K.learning_phase()], o)
    
    
class DFlatten(DLayer):
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


class DInput(DLayer):
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


class VisualModel():
    layers: List[DLayer] = []

    def __init__(self, model: Model
                     , layer_name = ''):
        self.save_image = save_image
        self.layer_name = layer_name

        for l in model.layers:
            if   isinstance(l, Convolution2D):
                self.layers.append(DConvolution2D(l))
                self.layers.append(DActivation(l))
            elif isinstance(l, MaxPooling2D):
                self.layers.append(DPooling(l))
            elif isinstance(l, Dense):
                self.layers.append(DDense(l))
                self.layers.append(DActivation(l))
            elif isinstance(l, Activation):
                self.layers.append(DActivation(l))
            elif isinstance(l, Flatten):
                self.layers.append(DFlatten(l))
            elif isinstance(l, InputLayer):
                self.layers.append(DInput(l))
            else:
                raise ValueError(f'Cannot handle this type of layer: \n{l.get_config()}')
            if l.name == layer_name:
                break

    
    def visualize(self, data
                      , top_n: int
                      , max_only = False
                      , save_img = lambda img, name: None ):
        up_data = self.up(data)
        for i, f, d in self.get_top_features(up_data, top_n, max_only):
            down_data = self.down(d)
            save_img(down_data.squeeze(), f'top{i}_feature{f}')


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
        

    
def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('image', help = 'Path of image to visualize')
    parser.add_argument('--layer_name', '-l', 
            action = 'store', dest = 'layer_name', 
            default = 'block5_conv3', help = 'Layer to visualize')
    parser.add_argument('--feature', '-f', 
            action = 'store', dest = 'feature', 
            default = 0, type = int, help = 'Feature to visualize')
    parser.add_argument('--mode', '-m', action = 'store', dest = 'mode', 
            choices = ['max', 'all'], default = 'max', 
            help = 'Visualize mode, \'max\' mode will pick the greatest \
                    activation in the feature map and set others to zero, \
                    \'all\' mode will use all values in the feature map')
    return parser


def load_image(path):
    img = Image.open(path).resize((224, 224))
    img_array = np.array(img)[np.newaxis, :].astype(np.float)
    return img_array


def save_image(data, path):
    data = data - data.min()
    data *= 1.0 / (data.max() + 1e-8)
    data = data[:, :, ::-1]
    uint8_data = (data * 255).astype(np.uint8)
    img = Image.fromarray(uint8_data, 'RGB')
    img.save(path)


def main():
    parser = argparser()
    args = parser.parse_args()
    image_path = args.image
    layer_name = args.layer_name
    visualize_mode = args.mode

    model = vgg16.VGG16(weights = 'imagenet', include_top = True)
    model.summary()
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    if not layer_name in layer_dict:
        print('Wrong layer name')
        sys.exit()

    # Load data and preprocess
    img = vgg16.preprocess_input(load_image(image_path))

    vis_model = VisualModel(model, layer_name)
    vis_model.visualize( img
                       , top_n = 5
                       , max_only = visualize_mode == 'max'
                       , save_img = lambda img, name: save_image(img, f'results/{layer_name}_{name}_{visualize_mode}.png')
                       )


if "__main__" == __name__:
    main()
