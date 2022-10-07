# Setup
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from glob import glob
TRAIN_DATA_FOLDER = '/home/est_posgrado_manuel.suarez/data/dogs-vs-cats/train5000'
TEST_DATA_FOLDER = '/home/est_posgrado_manuel.suarez/data/dogs-vs-cats/train1000'
# train_dog_files = np.array(glob(os.path.join(DATA_FOLDER, 'dog.*.jpg')))
# train_cat_files = np.array(glob(os.path.join(DATA_FOLDER, 'cat.*.jpg')))
train_dogs_files = np.array(glob(os.path.join(TRAIN_DATA_FOLDER, 'dog.*.jpg')))
train_cats_files = np.array(glob(os.path.join(TRAIN_DATA_FOLDER, 'cat.*.jpg')))
test_dogs_files = np.array(glob(os.path.join(TEST_DATA_FOLDER, 'dog.*.jpg')))
test_cats_files = np.array(glob(os.path.join(TEST_DATA_FOLDER, 'cat.*.jpg')))

train_dogs = tf.data.Dataset.list_files(train_dogs_files)
train_cats = tf.data.Dataset.list_files(train_cats_files)
test_dogs = tf.data.Dataset.list_files(test_dogs_files)
test_cats = tf.data.Dataset.list_files(test_cats_files)

# Define the standard image size.
orig_img_size = (286, 286)
# Size of the random crops to be used during training.
input_img_size = (28, 28, 3)
# Weights initializer for the layers.
kernel_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
# Gamma initializer for instance normalization.
gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

batch_size = 1

def normalize_img(img):
    img = tf.cast(img, dtype=tf.float32)
    # Map values in the range [-1, 1]
    return (img / 127.5) - 1.0


def preprocess_train_image(img):
    # Load image
    img = tf.io.read_file(img)
    # Decode
    img = tf.image.decode_jpeg(img)
    # Random flip
    img = tf.image.random_flip_left_right(img)
    # Resize to the original size first
    #img = tf.image.resize(img, [*orig_img_size])
    image = tf.image.resize(img, [286, 286],
                            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # Random crop to 256X256
    img = tf.image.random_crop(img, size=[28, 28, 3])
    # Normalize the pixel values in the range [-1, 1]
    img = normalize_img(img)
    return img


def preprocess_test_image(img):
    # Load image
    img = tf.io.read_file(img)
    # Decode
    img = tf.image.decode_jpeg(img)
    # Only resizing and normalization for the test images.
    img = tf.image.resize(img, [input_img_size[0], input_img_size[1]])
    img = normalize_img(img)
    return img

# Create datasets
autotune = tf.data.AUTOTUNE

# Apply the preprocessing operations to the training data
TRAIN_BUFFER_SIZE = len(train_dogs_files)
train_dogs = (
    train_dogs.map(preprocess_train_image, num_parallel_calls=autotune)
    .cache()
    .shuffle(TRAIN_BUFFER_SIZE)
    .batch(batch_size)
)
train_cats = (
    train_cats.map(preprocess_train_image, num_parallel_calls=autotune)
    .cache()
    .shuffle(TRAIN_BUFFER_SIZE)
    .batch(batch_size)
)

# Apply the preprocessing operations to the test data
TEST_BUFFER_SIZE = len(test_dogs_files)
test_dogs = (
    test_dogs.map(preprocess_test_image, num_parallel_calls=autotune)
    .cache()
    .shuffle(TEST_BUFFER_SIZE)
    .batch(batch_size)
)
test_cats = (
    test_cats.map(preprocess_test_image, num_parallel_calls=autotune)
    .cache()
    .shuffle(TEST_BUFFER_SIZE)
    .batch(batch_size)
)

train_dataset = tf.data.Dataset.zip((train_dogs, train_cats))
test_dataset = tf.data.Dataset.zip((test_dogs, test_cats))

# Sampling Layer
class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# Encoder
latent_dim = 2

encoder_inputs = keras.Input(shape=(28, 28, 1))
x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Flatten()(x)
x = layers.Dense(16, activation="relu")(x)
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
encoder.summary()
# Decoder
latent_inputs = keras.Input(shape=(latent_dim,))
x = layers.Dense(7 * 7 * 64, activation="relu")(latent_inputs)
x = layers.Reshape((7, 7, 64))(x)
x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
decoder.summary()

# VAE
class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        print(data)
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            print(reconstruction)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

# Training
vae = VAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Adam())
vae.fit(test_dogs, epochs=30, batch_size=128)

# Plot latent space
import matplotlib.pyplot as plt


def plot_latent_space(vae, n=30, figsize=15):
    # display a n*n 2D manifold of digits
    digit_size = 28
    scale = 1.0
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = vae.decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[
                i * digit_size : (i + 1) * digit_size,
                j * digit_size : (j + 1) * digit_size,
            ] = digit

    plt.figure(figsize=(figsize, figsize))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap="Greys_r")
    plt.savefig("figure1.png")


plot_latent_space(vae)