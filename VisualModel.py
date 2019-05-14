#!/usr/bin/env python

import numpy as np
import math
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
from PIL import Image


def reconstruct_image(img_arr):
    img_arr = np.squeeze(img_arr)
    img_arr = img_arr - img_arr.min()
    img_arr *= 1.0 / (img_arr.max() + 1e-8)
    uint8_data = (img_arr * 255).astype(np.uint8)
    return Image.fromarray(uint8_data, 'RGB')


class VisLayer:
    def __init__(self, layer):
        self.layer = layer
        self.name = layer.name
        self.up_func = lambda data: data
        self.down_func = lambda data: data

    def up(self, data):
        up_data = self.up_func([data])
        if isinstance(up_data, list) and len(up_data) == 1:
            up_data = up_data[0]
        return up_data

    def down(self, data):
        down_data = self.down_func([data])
        if isinstance(down_data, list) and len(down_data) == 1:
            down_data = down_data[0]
        return down_data


class VisConv2D(VisLayer):
    def __init__(self, layer: Conv2D):
        super().__init__(layer)

        W, b = layer.get_weights()
        i = Input(shape=layer.input_shape[1:])
        u = Conv2D(
            filters=layer.filters,
            kernel_size=layer.kernel_size,
            strides=layer.strides,
            dilation_rate=layer.dilation_rate,
            padding=layer.padding,
            kernel_initializer=tf.constant_initializer(W),
            bias_initializer=tf.constant_initializer(b),
        )
        self.up_func = K.function([i], [u(i)])

        # Flip each filter vertically and horizontally
        W_flipped = W[::-1, ::-1, :, :]
        # Ensure the reconstructed image has correct number of channels
        W_t = np.moveaxis(W_flipped, 2, 3)
        b_t = np.zeros(W_t.shape[3])
        o = Input(shape=layer.output_shape[1:])
        d = Conv2D(
            filters=W_t.shape[-1],
            kernel_size=W_t.shape[:2],
            strides=layer.strides,
            dilation_rate=layer.dilation_rate,
            padding=layer.padding,
            kernel_initializer=tf.constant_initializer(W_t),
            bias_initializer=tf.constant_initializer(b_t),
        )
        self.down_func = K.function([o], [d(o)])


class VisDense(VisLayer):
    def __init__(self, layer: Dense):
        super().__init__(layer)

        weights = layer.get_weights()
        W = weights[0]
        b = weights[1]

        # Set up_func for DDense
        i = Input(shape=layer.input_shape[1:])
        o = Dense(units=layer.output_shape[1])(i)
        o.weights = [W, b]
        self.up_func = K.function([i], [o])

        # Transpose W and set down_func for DDense
        W = W.transpose()
        self.input_shape = layer.input_shape
        self.output_shape = layer.output_shape
        b = np.zeros(self.input_shape[1])
        flipped_weights = [W, b]
        i = Input(shape=self.output_shape[1:])
        o = Dense(units=self.input_shape[1])(i)
        o.weights = flipped_weights
        self.down_func = K.function([i], [o])


class VisMaxPooling2D(VisLayer):
    def __init__(self, layer: MaxPooling2D):
        super().__init__(layer)

        i = Input(shape=layer.input_shape[1:])
        pool = MaxPooling2D(
            pool_size=layer.pool_size,
            strides=layer.strides,
            padding=layer.padding,
            data_format='channels_last'
        )
        self._maxpool = K.function([i], [pool(i)])

        o = Input(shape=layer.output_shape[1:])
        ups_factor = int(layer.input_shape[1] / layer.output_shape[1])
        ups = UpSampling2D(size=(ups_factor, ups_factor))
        self._upsample = K.function([o], [ups(o)])

        self._switches = np.zeros((1, *layer.input_shape[1:]))

    def up(self, data):
        up_data = self._maxpool([data])[0]
        self._switches = data == self._upsample([up_data])[0]
        return up_data

    def down(self, data):
        return self._upsample([data])[0] * self._switches


class VisActivation(VisLayer):
    def __init__(self, layer: Activation, linear=False):
        super().__init__(layer)
        self.linear = linear
        self.activation = layer.activation
        i = K.placeholder(shape=layer.output_shape)

        a = self.activation(i)
        # According to the original paper,
        # In forward pass and backward pass, do the same activation(relu)
        self.up_func = K.function([i], [a])
        self.down_func = K.function([i], [a])


class VisFlatten(VisLayer):
    def __init__(self, layer: Flatten):
        super().__init__(layer)
        self.shape = layer.input_shape[1:]
        self.up_func = K.function([layer.input], [layer.output])

    def up(self, data):
        # Flatten 2D input into 1D output
        return self.up_func([data])[0]

    def down(self, data):
        # Reshape 1D input into 2D output
        new_shape = [data.shape[0]] + list(self.shape)
        return np.reshape(data, new_shape)


class VisInput(VisLayer):
    def __init__(self, layer: Input):
        super().__init__(layer)

    def up(self, data):
        return data

    def down(self, data):
        return data


class VisModel:
    layers: List[VisLayer] = []

    def __init__(self, model: Model, layer_name):
        self.layer_name = layer_name
        self.layer_info = []
        self.receptive_field_info = dict()

        for l in model.layers:
            if isinstance(l, Conv2D):
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
                img_size = l.input_shape[1]
                self.layer_info.append([img_size, 1, 1, 0.5])
                self.layers.append(VisInput(l))
            else:
                raise ValueError(f'Cannot handle this type of layer: \n{l.get_config()}')

            # Compute layer info
            if isinstance(l, Conv2D) or isinstance(l, MaxPooling2D):
                filter_params = self.get_filter_params(l)
                current_layer_info = self.compute_layer_info(filter_params, self.layer_info[-1])
                self.layer_info.append(current_layer_info)
                self.receptive_field_info[l.name] = current_layer_info

            if l.name == layer_name:
                break

    def visualize(self, data, top_n: int, max_only=False, save_img=lambda img, name: None):
        up_data = self.up(data)
        for rank, feat, d in self.get_top_features(up_data, top_n, max_only):
            down_data = self.down(d)
            save_img(np.array(down_data).squeeze(), rank + 1, feat)

    def up(self, data):
        up_data = self.layers[0].up(data)
        for j in range(1, len(self.layers)):
            up_data = self.layers[j].up(up_data)
        return up_data

    def down(self, data):
        down_data = self.layers[-1].down(data)
        for j in reversed(range(len(self.layers[:-1]))):
            down_data = self.layers[j].down(down_data)

        return down_data

    def visualise_cropped(self, preprocessed_img_data, original_img):
        up_data = self.up(preprocessed_img_data)

        result = list(self.get_top_features(up_data, n=1, max_only=True))
        rank, top_filter_idx, feature_maps = result[0]
        down_data = self.down(feature_maps)

        # Find crop region
        top_feature_map = feature_maps[:, :, :, top_filter_idx]
        feature_idx = top_feature_map.argmax()
        recp_info = self.receptive_field_info[self.layers[-1].layer.name]
        output_size, jump, receptive_size, start = recp_info
        center_point = int((feature_idx*jump + start*jump) % 224)
        crop_pos = center_point - int(receptive_size / 2)
        crop_box = (crop_pos, crop_pos, crop_pos + receptive_size, crop_pos + receptive_size)

        reconstructed_img = reconstruct_image(down_data)

        cropped_reconstruction = reconstructed_img.crop(crop_box)
        cropped_original = original_img.crop(crop_box)
        return cropped_reconstruction, cropped_original

    def get_top_features(self, data, n: int, max_only=False):
        ''' Returns iterable yielding tuples of (ranking, feature_index, output_data)'''

        # Find the largest activation for each feature map
        f_max = np.array(list(map(
            lambda f: data[..., f].max(),
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

    def compute_layer_info(self, layer_filter_params, prev_layer_info):
        n_in = prev_layer_info[0]
        j_in = prev_layer_info[1]
        r_in = prev_layer_info[2]
        start_in = prev_layer_info[3]
        k = layer_filter_params[0]
        s = layer_filter_params[1]
        p = layer_filter_params[2]

        n_out = math.floor((n_in - k + 2 * p) / s) + 1
        actualP = (n_out - 1) * s - n_in + k
        pR = math.ceil(actualP / 2)
        pL = math.floor(actualP / 2)

        j_out = j_in * s
        r_out = r_in + (k - 1) * j_in
        start_out = start_in + ((k - 1) / 2 - pL) * j_in
        return n_out, j_out, r_out, start_out

    def down_bound(self, up_data):
        top_features = list(self.get_top_features(up_data, n=1, max_only=True))
        rank, top_feature_idx, feature_maps = top_features[0]
        top_feature_map = feature_maps[:, :, :, top_feature_idx]

        prev_layer = self.layers[-2].layer

        # Compute the filter size and padding
        filter_size, stride, padding = self.get_filter_params(prev_layer)
        batch, width, height = top_feature_map.shape
        highest_act_index = top_feature_map.argmax()

        print(f'Filter Index: {top_feature_idx}')
        print(f'Highest activation index: {highest_act_index}')

        print('  Width: {} Height: {}, Filter'.format(width, height))

        bbox = self.compute_bounding_box(
            width=width,
            height=height,
            filter_size=filter_size,
            padding=padding,
            index=highest_act_index
        )
        print('Found bounding box: {}'.format(bbox))

        start_index = bbox['start_index']
        end_index = bbox['end_index']

        for i in range(len(self.layers)-3, 0, -1):
            vis_layer = self.layers[i]
            if isinstance(vis_layer, VisConv2D) or isinstance(vis_layer, VisMaxPooling2D):
                layer = vis_layer.layer
                print('Processing: {}'.format(layer.name, i))
                filter_size, stride, padding = self.get_filter_params(layer)
                batch, width, height, channels = layer.output_shape
                bbox_start = self.compute_bounding_box(
                    width=width,
                    height=height,
                    filter_size=filter_size,
                    padding=padding,
                    index=start_index
                )

                print(' Start index: {}'.format(bbox_start))
                print('  Width: {} Height: {}'.format(width, height))

                bbox_end = self.compute_bounding_box(
                    width=width,
                    height=height,
                    filter_size=filter_size,
                    padding=padding,
                    index=end_index
                )

                print(' End index:   {}'.format(bbox_end))
                start_index = bbox_start['start_index']
                end_index = bbox_end['end_index']

            elif isinstance(vis_layer, VisInput):
                print('Reach the input layer: {}'.format((start_index, end_index)))

        return bbox

    def get_filter_params(self, layer):
        conf = layer.get_config()
        if 'kernel_size' in conf:
            pool_size = conf['kernel_size']
        elif 'pool_size' in conf:
            pool_size = conf['pool_size']
        else:
            raise Exception('Unknown layer: {}'.format(conf))
        assert (pool_size[0] == pool_size[1])
        pool_size = pool_size[0]

        stride = conf['strides']
        assert (stride[0] == stride[1])
        stride = stride[0]

        padding = conf['padding']
        if padding == 'same':
            padding = int((pool_size - 1) / 2)
        elif padding == 'valid':
            padding = 0
        else:
            raise Exception('Unknown padding type: {}'.format(padding))
        return pool_size, stride, padding

    def compute_bounding_box(self, width, height, filter_size, padding, index):
        row_idx = int(np.ceil((index + 1) / width))
        column_idx = index % height

        rf_row_start_idx = row_idx
        rf_column_start_idx = column_idx
        rf_row_end_idx = rf_row_start_idx + filter_size - 1
        rf_column_end_idx = rf_column_start_idx + filter_size - 1

        rf_row_start_idx = max(0, rf_row_start_idx - padding)
        rf_column_start_idx = max(0, rf_column_start_idx - padding)
        rf_row_end_idx = min(width - 1, rf_row_end_idx - padding)
        rf_column_end_idx = min(height - 1, rf_column_end_idx - padding)

        start_index = width*(rf_row_start_idx + 1) - (width - rf_column_start_idx)
        end_index = width * (rf_row_end_idx + 1) - (width - rf_column_end_idx)

        return {
            'start_index': start_index,
            'end_index': end_index,
            'start_row': rf_row_start_idx,
            'start_column': rf_column_start_idx,
            'end_row': rf_row_end_idx,
            'end_column': rf_column_end_idx
        }
