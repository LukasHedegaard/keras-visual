from tensorflow.keras.applications import vgg16
from PIL import Image
import numpy as np
from VisualModel import VisModel


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
    # Params
    vis_layer_name = 'block3_conv3'
    vis_mode = 'all'
    image_path = 'husky.jpg'

    # Load model
    model = vgg16.VGG16(weights = 'imagenet', include_top = True)
    model.summary()
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    if not vis_layer_name in layer_dict:
        raise ValueError('Wrong layer name: ' + vis_layer_name)

    # Visualize!
    img = vgg16.preprocess_input(load_image(image_path))

    vis_model = VisModel(model, vis_layer_name)
    vis_model.visualize( img
                       , top_n = 3
                       , max_only = vis_mode == 'max'
                       , save_img = lambda img, rank, feat: save_image(img, f'results/{vis_layer_name}_top{rank}_feature{feat}_{vis_mode}.png')
                       )


if "__main__" == __name__:
    main()