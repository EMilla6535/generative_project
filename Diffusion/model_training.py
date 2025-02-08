import tensorflow as tf
import keras
import os
from shutil import make_archive
from zipfile import ZipFile
from PIL import Image, ImageDraw

from model import DiffusionModel
from dataset_loader import load_dataset

BASE_PATH = '/path/to/save/results'
DATASET_PATH = 'path/to/dataset'

if not os.path.exists(BASE_PATH):
    os.makedirs(BASE_PATH + '/result_images')

class SampleCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Every 10 epochs
        # Generate 10 samples
        # And save them
        if (epoch + 1) % 10 == 0:
            n = 10
            rng = tf.random.Generator.from_non_deterministic_state()
            x = rng.normal(shape=[n, self.model.img_size, self.model.img_size, 3], mean=0.0, stddev=1.0, dtype='float32')
            result = self.model.predict(x, batch_size=n)
            for i in range(n):
                tf.keras.utils.save_img(f'{BASE_PATH}/result_images/{epoch}_result_{i}.jpg', result[i], data_format="channels_last", scale=True)

epochs = 500
model = DifussionModel(x_shape=(64, 64, 3, ), t_shape=(None, ))

mc = tf.keras.callbacks.ModelCheckpoint('saved_model.keras', monitor='loss', save_best_only=True, mode='min')
sc = SampleCallback()

train_dataset = load_dataset(DATASET_PATH)

optimizer = tf.keras.optimizers.AdamW(learning_rate=3e-4)
mse_loss = tf.keras.losses.MeanSquaredError()
model.compile(optimizer=optimizer, loss=mse_loss, metrics=['mse'])

model.fit(x=train_dataset, epochs=epochs, callbacks=[mc, sc])

# Zip results
filename = "saved_results"
directory = BASE_PATH + '/result_images'
make_archive(filename, "zip", directory)
print("Finished!")