from tensorflow.keras.applications import vgg16
from tensorflow.keras.layers import Conv2D
import os
from PIL import Image
import numpy as np
import VisualModel
from importlib import reload
import argparse as ap
from pathlib import Path
from imagenet_labels import imagenet_labels
from tqdm import tqdm


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
        raise ap.ArgumentTypeError(f'Image dir "{image_path}" does not exist.')
    return image_path


def parse_args():
    parser = ap.ArgumentParser(description='Creates feature visualisations.')
    parser.add_argument('--image-dir', '-i',
                        type=check_image_path,
                        required=False,
                        default='images',
                        help=f'The path of the image. Default: images')
    return parser.parse_args()


def generate_visualisations(model, layer_name, preprocessed_image, prefix, top_n, pbar):
    output_dir = Path(f'output/top{top_n}/{layer_name}')
    output_dir.mkdir(parents=True, exist_ok=True)

    reload(VisualModel)  # Reload model to avoid strange error
    vis_model = VisualModel.VisModel(model, layer_name)

    # Forward pass
    feature_maps = vis_model.up(preprocessed_image)

    for rank, filter_idx, d in vis_model.get_top_features(feature_maps, top_n, True):
        down_data = vis_model.down(d)
        projected_img_data = down_data.squeeze()
        image_path = output_dir / f'{prefix}_{layer_name}_top{rank}_filter{filter_idx}.jpg'
        save_image(projected_img_data, image_path)
        pbar.update()


def main():
    args = vars(parse_args())
    print(f'Running with args:\n  {args}')

    image_dir = Path(args['image_dir'])

    print('Loading the model...')
    model = vgg16.VGG16(weights='imagenet', include_top=True)

    conv_layers = [l for l in model.layers if isinstance(l, Conv2D)]
    top_n = 10
    total_steps = len(conv_layers) * top_n

    image_paths = [image_dir / fn for fn in os.listdir(image_dir)]
    for image_path in image_paths:
        original_image = Image.open(image_path).resize((224, 224))
        img_array = np.array(original_image)[np.newaxis, :].astype(np.float)
        preprocessed_img = vgg16.preprocess_input(img_array)

        # Make prediction
        prediction = model.predict(preprocessed_img)
        pred_label_index = np.argmax(prediction, axis=1)[0]
        predicted_class = imagenet_labels[pred_label_index]
        print(f'Prediction for image {image_path} is "{predicted_class}"')

        # Get image name without file extension
        prefix = image_path.name.split('.')[0]

        print(f'Visualising features ...')

        with tqdm(total=total_steps) as pbar:
            for layer in conv_layers:
                generate_visualisations(model, layer.name, preprocessed_img, prefix, top_n, pbar)

    print('Done!')


if "__main__" == __name__:
    main()
