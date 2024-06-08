"""
Utilitary code to create a dataset for training
"""
import tensorflow as tf
import glob
import os
from shutil import make_archive
from zipfile import ZipFile

def load_image(filename, img_size):
    """
    - Load a single .jpg image.
    - Resize image to img_size.
    - Rescale values to be in range [0, 1].
    - Return values as list.
    """
    raw = tf.io.read_file(filename)
    image = tf.io.decode_jpeg(raw, channels=3)
    image = tf.image.resize(image, [img_size, img_size], antialias=True)
    rescaling_layer = tf.keras.layers.Rescaling(scale=1./255)
    image = rescaling_layer(image)
    return image.numpy().tolist()

filepath = r"/path/to/images"
y_image_size = 256

y_data = []
for i, filename in enumerate(glob.glob(filepath + '/*.jpg')):
    """
    - Load every .jpg image in [filepath] and append it
    in a list.
    """
    y_image = load_image(filename, y_image_size)
    y_data.append(y_image)
            
img_dataset = tf.data.Dataset.from_tensor_slices(y_data)
img_dataset.save(f"dataset/folder")