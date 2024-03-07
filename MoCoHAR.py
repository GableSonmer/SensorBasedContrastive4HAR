import numpy as np
import tensorflow as tf
from keras.src.callbacks import EarlyStopping, CSVLogger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Sequential
from tensorflow.keras import Model
from tensorflow.keras.layers import *
from tqdm import tqdm

from DeepConvLSTM import get_DCL
import time
from tensorflow.keras.metrics import *
from Augment import *


class MoCo(Model):
    def __init__(self, base_encoder, dim=128, K=8192, m=0.999, T=0.07, mlp=True, n_timesteps=128, n_features=9):
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T
        self.dim = dim
        self.encoder_q = base_encoder(n_timesteps=n_timesteps, n_features=n_features)
        self.encoder_k = base_encoder(n_timesteps=n_timesteps, n_features=n_features)

        if mlp:
            self.encoder_q = Sequential([self.encoder_q, Dense(dim * 2), ReLU(), Dense(dim)])
            self.encoder_k = Sequential([self.encoder_k, Dense(dim * 2), ReLU(), Dense(dim)])

        for i in range(len(self.encoder_q.layers)):
            self.encoder_k.get_layer(index=i).set_weights(
                self.encoder_q.get_layer(index=i).get_weights())

        self.encoder_k.trainable = False

        _queue = np.random.normal(size=(dim, self.K))
        _queue /= np.linalg.norm(_queue, axis=0)
        self.queue = self.add_weight(
            name='queue',
            shape=(dim, self.K),
            initializer=tf.keras.initializers.Constant(_queue),
            trainable=False)

        self.queue_ptr = self.add_weight(name='queue_ptr', shape=(1,), initializer=tf.keras.initializers.zeros(),
                                         trainable=False)

    def reset_queue(self):
        tf.compat.v1.assign(self.queue_ptr[0], 0)
        _queue = np.random.normal(size=(self.dim, self.K))
        _queue /= np.linalg.norm(_queue, axis=0)
        tf.compat.v1.assign(self.queue, _queue)

    def _momentum_update_key_encoder(self):
        """
            Momentum update of the key encoder
        """
        for i in range(len(self.encoder_q.weights)):
            tf.compat.v1.assign(self.encoder_k.weights[i],
                                tf.add(self.encoder_k.weights[i] * tf.constant(self.m, dtype=tf.float32),
                                       self.encoder_q.weights[i] * tf.constant((1. - self.m), dtype=tf.float32)))

    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        # assert self.K % batch_size == 0,'Make sure batchsize is a factor of K'

        tf.compat.v1.assign(self.queue[:, ptr:ptr + batch_size], tf.transpose(keys))

        ptr = (ptr + batch_size) % self.K  # move pointer

        tf.compat.v1.assign(self.queue_ptr[0], ptr)

    def call(self, input):

        batch_size = input[0].shape[0]
        # compute query features
        q = self.encoder_q(input[0])  # queries: NxC
        q = tf.math.l2_normalize(q, axis=1)

        # compute key features
        # with tf.no_gradient():  # no gradient to keys
        self._momentum_update_key_encoder()  # update the key encoder

        k = self.encoder_k(input[1])  # keys: NxC
        k = tf.math.l2_normalize(k, axis=1)
        k = tf.stop_gradient(k)

        # compute logits
        # Einstein sum is more intuitive
        l_pos = tf.reshape(tf.einsum('nc,nc->n', q, k), (-1, 1))  # nx1
        l_neg = tf.einsum('nc,ck->nk', q, self.queue)  # nxK
        logits = tf.concat([l_pos, l_neg], axis=1)  # nx(1+k)
        logits = logits * (1 / self.T)

        # labels: positive key indicators
        labels = tf.zeros(batch_size, dtype=tf.int64)  # n

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels


def evaluate(model_cl, reserve_layer, outputs, method='linear'):
    if method == 'linear':
        model = Model(model_cl.layers[0].input, model_cl.layers[reserve_layer].output, trainable=False)
    else:
        model = Model(model_cl.layers[0].input, model_cl.layers[reserve_layer].output, trainable=True)
    return Sequential([model, Dense(outputs, activation='softmax')])


def main(x_data, y_data, transform1, transform2):
    # IMPORTANT: clear the session before training
    tf.keras.backend.clear_session()

    n_timesteps, n_features, n_outputs = x_data.shape[1], x_data.shape[2], y_data.shape[1]

    model_cl = MoCo(get_DCL, n_timesteps=n_timesteps, n_features=n_features)

    batch_size = 1024
    epochs = 200
    loss_function = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.optimizers.Adam(learning_rate=0.001)
    for epoch in tqdm(range(epochs)):
        loss_epoch = []
        train_loss_dataset = tf.data.Dataset.from_tensor_slices(x_data).shuffle(len(x_data),
                                                                                reshuffle_each_iteration=True).batch(
            batch_size)
        for x in train_loss_dataset:
            # xis = resampling_fast_random(x)  # Select the augmentation method used
            # xjs = x  # Select the augmentation method used
            xis, xjs = transform1(x), transform2(x)
            x_cat = [xis, xjs]
            with tf.GradientTape() as tape:
                logits, labels = model_cl(x_cat)
                loss = tf.reduce_mean(loss_function(labels, logits))
            gradients = tape.gradient(loss, model_cl.encoder_q.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model_cl.encoder_q.trainable_variables))
            loss_epoch.append(loss)
        tf.compat.v1.assign(model_cl.queue_ptr[0], 0)
        if (epoch + 1) % 50 == 0:
            tqdm.write(f'Epoch [{epoch}/{epochs}], Loss: {np.mean(loss_epoch)}')

    timestamp = time.time()
    model_cl.encoder_q.save('contrastive_model/moco/MoCoHAR_' + '_' + str(timestamp) + '.h5')

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.99)
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

    linear_model = evaluate(model_cl.encoder_q, -4, n_outputs, 'linear')
    linear_model.compile(
        loss="categorical_crossentropy",
        metrics=[Precision(), F1Score(threshold=0.5, average='micro')],
        optimizer=tf.keras.optimizers.Adam(0.01)
    )
    history_linear = linear_model.fit(
        x_train, y_train,
        epochs=200,
        batch_size=50,
        validation_data=(x_test, y_test),
        shuffle=True,
        callbacks=[early_stopping,
                   CSVLogger(
                       f'contrastive_model/moco/logger/log_linear_{transform1.__name__}_{transform2.__name__}.csv')]
    )
    # print('linear best accuracy: {}'.format(np.max(history_linear.history['val_f1_score'])))
    linear_best = np.max(history_linear.history['val_f1_score'])

    fine_model = evaluate(model_cl.encoder_q, -4, n_outputs, 'fine')
    fine_model.compile(
        loss="categorical_crossentropy",
        metrics=[Precision(), F1Score(threshold=0.5, average='micro')],
        optimizer=tf.keras.optimizers.Adam(0.0005)
    )
    history_fine = fine_model.fit(
        x_train, y_train,
        epochs=200,
        batch_size=50,
        validation_data=(x_test, y_test),
        shuffle=True,
        callbacks=[early_stopping,
                   CSVLogger(
                       f'contrastive_model/moco/logger/log_linear_{transform1.__name__}_{transform2.__name__}.csv')]
    )
    print('fine best accuracy: {}'.format(np.max(history_fine.history['val_f1_score'])))
    fine_best = np.max(history_fine.history['val_f1_score'])

    return linear_best, fine_best
