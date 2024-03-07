import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.metrics import *
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger
from tensorflow.keras.optimizers import Adam
from DeepConvLSTM import get_DCL
import time
from tqdm import tqdm


def attach_projection_head(backbone, dim1=256, dim2=128, dim3=50):
    return Sequential([backbone, Dense(dim1), ReLU(), Dense(dim2), ReLU(), Dense(dim3)])


def contrastive_loss(out, out_aug, batch_size, hidden_norm=True, temperature=1.0, weights=1.0):
    LARGE_NUM = 1e9
    entropy_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    h1 = out
    h2 = out_aug
    if hidden_norm:
        h1 = tf.math.l2_normalize(h1, axis=1)
        h2 = tf.math.l2_normalize(h2, axis=1)

    labels = tf.range(batch_size)
    masks = tf.one_hot(tf.range(batch_size), batch_size)

    logits_aa = tf.matmul(h1, h1, transpose_b=True) / temperature
    logits_aa = logits_aa - masks * LARGE_NUM
    logits_bb = tf.matmul(h2, h2, transpose_b=True) / temperature
    logits_bb = logits_bb - masks * LARGE_NUM

    logits_ab = tf.matmul(h1, h2, transpose_b=True) / temperature
    logits_ba = tf.matmul(h2, h1, transpose_b=True) / temperature

    loss_a = entropy_function(labels, tf.concat([logits_ab, logits_aa], 1), sample_weight=weights)
    loss_b = entropy_function(labels, tf.concat([logits_ba, logits_bb], 1), sample_weight=weights)
    loss = loss_a + loss_b
    return loss


def train_step(xis, xjs, model, optimizer, temperature):
    with tf.GradientTape() as tape:
        zis = model(xis)
        zjs = model(xjs)
        batch_size = len(xis)
        loss = contrastive_loss(zis, zjs, batch_size, hidden_norm=True, temperature=temperature)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


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
    backbone = get_DCL(n_timesteps, n_features)
    model_cl = attach_projection_head(backbone)

    batch_size = 1024
    epochs = 200
    temperature = 0.1
    optimizer = Adam(0.001)

    # model_cl.save('contrastive_model/SimCLRHAR_' + str(timestamp) + '.h5')

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.90)
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

    # contrastive learning
    for epoch in tqdm(range(epochs)):
        loss_epoch = []
        train_loss_dataset = tf.data.Dataset.from_tensor_slices(x_data).shuffle(len(y_data),
                                                                                reshuffle_each_iteration=True).batch(
            batch_size)
        for x in train_loss_dataset:
            # xis = resampling_fast_random(x)  # Select the augmentation method used
            # xjs = noise(x)  # Select the augmentation method used
            xis, xjs = transform1(x), transform2(x)
            loss = train_step(xis, xjs, model_cl, optimizer, temperature=temperature)
            loss_epoch.append(loss)
        if (epoch + 1) % 50 == 0:
            tqdm.write(f'Epoch [{epoch}/{epochs}], Loss: {np.mean(loss_epoch)}')
    timestamp = time.time()

    # linear evaluation
    linear_model = evaluate(model_cl, -6, n_outputs, 'linear')
    linear_model.compile(
        loss="categorical_crossentropy",
        metrics=[Precision(), F1Score(threshold=0.5, average='micro')],
        optimizer=tf.keras.optimizers.Adam(0.01)
    )
    history_linear = linear_model.fit(
        x_train, y_train,
        epochs=100,
        batch_size=50,
        validation_data=(x_test, y_test),
        shuffle=True,
        callbacks=[early_stopping,
                   CSVLogger(f'contrastive_model/simclr/logger/log_linear_{transform1.__name__}_{transform2.__name__}.csv')]
    )

    linear_best_acc = np.max(history_linear.history['val_precision'])
    # print(f'linear best accuracy: {linear_best_acc * 100:.2f}')

    # fine-tuning
    fine_model = evaluate(model_cl, -6, n_outputs, 'fine')
    fine_model.compile(
        loss="categorical_crossentropy",
        metrics=[Precision(), F1Score(threshold=0.5, average='micro')],
        optimizer=tf.keras.optimizers.Adam(0.0005)
    )
    history_fine = fine_model.fit(
        x_train, y_train,
        epochs=100,
        batch_size=50,
        validation_data=(x_test, y_test),
        shuffle=True,
        callbacks=[early_stopping, CSVLogger(
            f'contrastive_model/simclr/logger/log_finetune_{transform1.__name__}_{transform2.__name__}.csv')]
    )

    fine_best_acc = np.max(history_fine.history['val_precision_1'])
    # print(f'fine best accuracy: {fine_best_acc * 100:.2f}')

    return linear_best_acc, fine_best_acc
