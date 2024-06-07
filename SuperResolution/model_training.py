import tensorflow as tf
import keras

import os
import glob
from shutil import make_archive
from zipfile import ZipFile

from model import DiffusionModel
from dataset_loader import load_dataset

# Check if result path exists
base_path = '/path/to/save/results'
if not os.path.exists(base_path):
    os.makedirs(base_path + '/result_images')

# Load sample images to test model
class SampleCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        """
        - Every 10 epochs:
            - Load some low resolution test images.
            - Rescale and upsample every image.
            - Set constant values for [s] noise step and [u] noise scale
            - Generate Gaussian noise and add it to the test images using
            the previous parameters
            - Use model.predict and save result
        """
        if (epoch + 1) % 10 == 0:
            test_img = []
            filepath = '/path/to/test/images'
            rescaling_layer = tf.keras.layers.Rescaling(scale=1./255)
            upsample_layer = tf.keras.layers.UpSampling2D(size=4, data_format='channels_last', interpolation='bicubic')
            for filename in glob.glob(filepath + '/*.jpg'):
                raw = tf.io.read_file(filename)
                image = tf.io.decode_jpeg(raw, channels=3)
                image = rescaling_layer(image)
                image = upsample_layer(tf.expand_dims(image, axis=0))
                test_img.append(image.numpy().tolist())
            test_img = tf.constant(test_img)
            test_img = tf.squeeze(test_img, axis=1)
            # Rescaled image, noise
            rng = tf.random.Generator.from_non_deterministic_state()
            
            s = tf.ones(test_img.shape[0], dtype='int32') * 2
            u = tf.ones(test_img.shape[0], dtype='float32') * 0.5
            s = tf.expand_dims(s, axis=-1)
            u = tf.expand_dims(u, axis=-1)

            noise = rng.normal(shape=test_img.shape, mean=0.0, stddev=1.0, dtype='float32')
            in_data = self.model.noise_inputs(test_img, s, u, noise)
            result = self.model.predict(in_data, batch_size=test_img.shape[0])
            for i in range(result.shape[0]):
                tf.keras.utils.save_img(f'{base_path}/result_images/{epoch}_result_{i}.jpg', result[i], data_format='channels_last', scale=True)

epochs = 100

mc = tf.keras.callbacks.ModelCheckpoint('super_res.weights.h5', monitor='loss', save_best_only=True, mode='min', save_weights_only=True)
sc = SampleCallback()

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
mse_loss = tf.keras.losses.MeanSquaredError()

train_dataset = load_dataset('/path/to/dataset')
"""
If first_train = False -> continue training using saved weights.
"""
first_train = True

if first_train:
    model = DiffusionModel(bot_blocks=3, pred_steps=50)
    model.compile(optimizer=optimizer, loss=mse_loss, metrics=['mse'])
    model.fit(train_dataset, epochs=epochs, callbacks=[mc, sc])
else:
    loaded_model = DiffusionModel(bot_blocks=3, pred_steps=70)
    loaded_model(normal, t_sample)
    loaded_model.load_weights("/path/to/saved/weights")
    loaded_model.compile(optimizer=optimizer, loss=mse_loss, metrics=['mse'])
    # Re-train
    loaded_model.fit(train_dataset, epochs=epochs, callbacks=[mc, sc])

filename = "sampled_results"
directory = base_path + '/result_images'
make_archive(filename, "zip", directory)
print("Finished!")