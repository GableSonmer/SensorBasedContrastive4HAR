"""
Variational Autoencoder for Human Activity Recognition
"""
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import h5py
import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
from sklearn.model_selection import train_test_split

files = ['self_collected.h5', 'mmfi.h5', 'that.h5']
idx = 0

# load data
path = os.path.join('datasets', files[idx])
with h5py.File(path, 'r') as hf:
    data = hf['data'][:]
    label = hf['label'][:]
    data = data.reshape(data.shape[0], 150, 4)

print('Loaded data from', path)
print('Data shape:', data.shape)
print('Label shape:', label.shape)


# vae model
class Sampling(layers.Layer):
    """使用（z_mean, z_log_var）进行采样的层，返回相应的z值。"""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def build_vae(input_shape=(150, 4), latent_dim=32):
    # 编码器
    encoder_inputs = layers.Input(shape=input_shape)
    x = layers.Flatten()(encoder_inputs)
    x = layers.Dense(512, activation='relu')(x)
    z_mean = layers.Dense(latent_dim, name='z_mean')(x)
    z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = models.Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder')

    # 解码器
    latent_inputs = layers.Input(shape=(latent_dim,))
    x = layers.Dense(512, activation='relu')(latent_inputs)
    x = layers.Dense(tf.math.reduce_prod(input_shape), activation='relu')(x)
    decoder_outputs = layers.Reshape(input_shape)(x)
    decoder = models.Model(latent_inputs, decoder_outputs, name='decoder')

    # VAE模型
    vae_outputs = decoder(encoder(encoder_inputs)[2])
    vae = models.Model(encoder_inputs, vae_outputs, name='vae')

    # 添加VAE损失
    # reconstruction_loss = tf.reduce_mean(
    #     tf.reduce_sum(
    #         tf.keras.losses.mean_squared_error(encoder_inputs, vae_outputs),
    #         axis=(1, 2)
    #     )
    # )
    reconstruction_loss = tf.reduce_mean(
        tf.keras.losses.mean_squared_error(tf.keras.layers.Flatten()(encoder_inputs),
                                           tf.keras.layers.Flatten()(vae_outputs))
    )

    kl_loss = -0.5 * tf.reduce_mean(
        tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
    )
    vae_loss = reconstruction_loss + kl_loss
    vae.add_loss(vae_loss)

    return encoder, decoder, vae


encoder, decoder, vae = build_vae()

# 编译VAE模型
vae.compile(
    optimizer='adam',
    metrics=['accuracy']
)

# 模型摘要
encoder.summary()
decoder.summary()
vae.summary()

x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=42)

# 训练模型
vae.fit(x_train, x_train, epochs=100, batch_size=32, validation_data=(x_test, x_test))
