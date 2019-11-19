from __future__ import absolute_import, division, print_function, unicode_literals

import datetime

import tensorflow as tf
import os
import numpy as np
import PIL.Image
import tensorflow_hub as hub
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/1')

BASE_DIR = os.path.abspath(os.curdir)
STYLE_DIR = os.path.join(BASE_DIR, "image_for_style")
name_dir_now = str(datetime.datetime.now().strftime("%H-%M_%d-%m-%Y"))
OUTPUT_DIR = os.path.join(BASE_DIR, f"output_images/{name_dir_now}")
IMAGES_DIR = os.path.join(BASE_DIR, "img")

try:
    os.mkdir(OUTPUT_DIR)
except:
    print("Directory exist")


def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)


def load_img(path_to_img):
    max_dim = 512
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img


def get_new_style_img(path_to_content_img, path_to_style_img, name="out.png"):
    content_img = load_img(path_to_content_img)
    style_img = load_img(path_to_style_img)
    stylized_image = hub_module(tf.constant(content_img), tf.constant(style_img))[0]
    new_image = tensor_to_image(stylized_image)
    new_image.save(name)


if __name__ == '__main__':

    files = [os.path.join(IMAGES_DIR, f) for f in os.listdir(IMAGES_DIR) if os.path.isfile(os.path.join(IMAGES_DIR, f))]
    styles = [os.path.join(STYLE_DIR, f) for f in os.listdir(STYLE_DIR) if
              os.path.isfile(os.path.join(STYLE_DIR, f))]

    for style in styles:
        for number, image in enumerate(files):
            print("Picture: № ", number)
            print("Name: ", image)
            print("-" * 50)
            output_name = os.path.join(OUTPUT_DIR, f"{style[-10:len(style)-4]}-{number}.png")
            get_new_style_img(image, style, output_name)
