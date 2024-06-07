import tensorflow as tf
import keras

"""
Dataset:
x = image_size -> 256
s = uniform(shape=[batch], minval=1, maxval=noise_steps, dtype='int32')
u_scale = uniform(shape[batch], minval=0.0, maxval=1.0, dtype='float32')
noise = normal(shape[batch, img_size, img_size, channels], mean=0.0, stddev=1.0)
"""
def random_generator(img_shape, s_shape, noise_steps):
    rng = tf.random.Generator.from_non_deterministic_state()
    s_uniform_gen = rng.uniform(shape=s_shape, minval=1, maxval=noise_steps, dtype='int32')
    u_uniform_gen = rng.uniform(shape=s_shape, minval=0.0, maxval=1.0, dtype='float32')
    normal_gen = rng.normal(shape=img_shape, mean=0.0, stddev=1.0, dtype='float32')
    for s, u, noise in zip(s_uniform_gen, u_uniform_gen, normal_gen):
        yield s, u, noise

def get_random_ds(image_size, n_items, noise_steps):
    random_ds = tf.data.Dataset.from_generator(random_generator,
                                               args=([n_items, image_size, image_size, 3], [n_items, 1], noise_steps),
                                               output_signature=(
                                                   tf.TensorSpec(shape=(1), dtype='int32'),
                                                   tf.TensorSpec(shape=(1), dtype='float32'),
                                                   tf.TensorSpec(shape=(image_size, image_size, 3), dtype='float32')))
    return random_ds.repeat(count=None)

"""
Note about dataset: This should be a saved dataset using tf.data.Dataset.save() method.
"""
# dataset_path = '/path/to/dataset' <- For testing
"""
load_dataset():
- Load image dataset. Note: The original dataset was too big, so I splitted it into four smaller parts.
* To do: Normalize values using mean=0.5, variance=0.25
- Get the total number of samples.
- Get random dataset containing:
    - s equivalent to noise steps.
    - u scaling value for noising image.
    - Gaussian noise.
- Join this two datasets.
"""
def load_dataset(dataset_path, batch_size=9, img_size=256, noise_steps=100):
    img_dataset = tf.data.Dataset.load(dataset_path)
    
    n_train = img_dataset.cardinality().numpy()
    rnd_dataset = get_random_ds(img_size, n_train, noise_steps)
    train_dataset = tf.data.Dataset.zip(img_dataset, rnd_dataset).shuffle(n_train).batch(batch_size).prefetch(batch_size)
    return train_dataset