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
import math
import os
from os import path
from pathlib import Path
from glob import glob
from PIL import Image, ImageFont, ImageDraw
import re


def load_filter_image(filter_img):
    orig_img_name = re.findall(r'(husky-\d{2})', filter_img)[0] + '.jpg'
    orig_img_path = Path('images') / orig_img_name
    background = Image.open(orig_img_path).convert('RGBA')
    foreground = Image.open(filter_img).convert('RGBA')
    foreground.putalpha(240)
    return Image.alpha_composite(background, foreground)


def load_filter_image_with_filter_index(filter_img):
    image = load_filter_image(filter_img)
    filter_index = re.findall(r'filter(\d+)', filter_img)[0]
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("arial.ttf", 20)
    draw.text((5, 200), f'#{filter_index}', (255, 255, 255), font=font)
    return image


def combine_horizontally(images, output_path):
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths) + len(widths) + 2
    max_height = max(heights) + 2

    new_im = Image.new('RGB', (total_width, max_height))
    x_offset = 1
    for im in images:
        new_im.paste(im, (x_offset, 1))
        x_offset += im.size[0] + 1
    new_im.save(output_path)
    return new_im


def visualise_filter(layer_name, filter_index):
    search_query = path.join('output', 'all', layer_name, f'*filter{filter_index}.jpg')
    image_paths = glob(search_query)
    filter_images = list(map(load_filter_image, image_paths))
    output_dir = Path('figures') / layer_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'{layer_name}_filter{filter_index}.jpg'
    return combine_horizontally(filter_images, output_path)


def main():
    print('Loading the model...')
    model = vgg16.VGG16(weights='imagenet', include_top=True)

    conv_layers = [l for l in model.layers if isinstance(l, Conv2D)]
    total_steps = np.sum([l.filters for l in conv_layers])

    with tqdm(total=total_steps) as pbar:
        for layer in conv_layers:
            for filter_idx in range(layer.filters):
                visualise_filter(layer.name, filter_idx)
                pbar.update()
    print('Done!')


if "__main__" == __name__:
    main()
