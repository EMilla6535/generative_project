import tensorflow as tf
import keras

def random_generator(img_shape, t_shape, noise_steps):
    rng = tf.random.Generator.from_non_deterministic_state()
    """
    IMPORTANT!!!
    For uniform(t values), shape must be [total_items, 1]
    For normal(noise), shape must be [total_items, W, H, C]
    """
    uniform_gen = rng.uniform(shape=t_shape, minval=1, maxval=1000, dtype='int32')
    normal_gen = rng.split(1)[0].normal(shape=img_shape, mean=0.0, stddev=1.0, dtype='float32')

    for t, noise in zip(uniform_gen, normal_gen):
        yield t, noise

def get_random_ds(image_size, n_items, noise_steps):
    random_ds = tf.data.Dataset.from_generator(random_generator,
                                               args=([n_items, image_size, image_size, 3], [n_items, 1], noise_steps),
                                               output_signature=(
                                                    tf.TensorSpec(shape=(1), dtype='int32'),
                                                    tf.TensorSpec(shape=(image_size, image_size, 3), dtype='float32')))
    return random_ds.repeat(count=None)

batch_size = 9
"""
Note about dataset: This should be a saved dataset using tf.data.Dataset.save() method.
"""
img_dataset_path = '/path/to/dataset'
"""
load_dataset():
- Load image dataset.
- Normalize values with mean=0.5 & variance=0.25
- Get the total number of samples.
- Get random dataset containing:
    - Gaussian noise.
    - t samples representing noise steps.
- Join this two datasets.
"""
def load_dataset(img_dataset_path, batch_size):
    img_dataset = tf.data.Dataset.load(img_dataset_path)
    img_dataset = img_dataset.map(lambda x: tf.keras.layers.Normalization(mean=0.5, variance=0.25, axis=None)(x))
    n_train = img_dataset.cardinality().numpy()
    rnd_dataset = get_random_ds(64, n_train, 1000)
    train_dataset = tf.data.Dataset.zip(img_dataset, rnd_dataset).shuffle(n_train).batch(batch_size).prefetch(batch_size)
    return train_dataset