from tensorflow.keras.applications import vgg16
from PIL import Image
import numpy as np
from VisualModel import VisModel
import argparse as ap
from pathlib import Path


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


def check_image_path(image_path):
    image_path = Path(image_path)
    if not image_path.exists():
        raise ap.ArgumentTypeError(f'Image path "{image_path}" does not exist.')
    return image_path


def parse_args():
    parser = ap.ArgumentParser(description='Creates feature visualisations.')
    parser.add_argument('--image-path', '-i',
                        type=check_image_path,
                        required=False,
                        default='images/husky-01.jpg',
                        help=f'The path of the image. Default: images/husky-01.jpg')
    parser.add_argument('--layer-name', '-l',
                        type=str,
                        required=False,
                        default='block3_conv3',
                        help=f'The name of layer to visualise the features for. Default: block3_conv3')
    parser.add_argument('--mode', '-m',
                        type=str,
                        required=False,
                        default='all',
                        help=f'Whether to show all features or the maximum feature. Default: all')
    parser.add_argument('--top-n', '-n',
                        type=int,
                        required=False,
                        default=3,
                        help=f'Number of top features to visualise. Default: 3')
    return parser.parse_args()


def main():
    args = vars(parse_args())
    print(f'Running with args:\n  {args}')

    vis_layer_name = args['layer_name']  # 'block3_conv3'
    vis_mode = args['mode']  # 'max'
    image_path = args['image_path']  # 'images/husky-01.jpg'

    print('Loading the model...')
    model = vgg16.VGG16(weights='imagenet', include_top=True)

    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    if vis_layer_name not in layer_dict:
        raise ValueError(f'Wrong layer name: {vis_layer_name}')

    print(f'Visualising features for {image_path}...')
    img = vgg16.preprocess_input(load_image(image_path))

    vis_model = VisModel(model, vis_layer_name)
    vis_model.visualize(img
                        , top_n=args['top_n']
                        , max_only=vis_mode == 'max'
                        , save_img=lambda img, rank, feat: save_image(img,
                                                                      f'results/{vis_layer_name}_top{rank}_feature{feat}_{vis_mode}.png')
                        )
    print('Done!')


if "__main__" == __name__:
    main()
