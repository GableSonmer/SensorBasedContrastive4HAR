import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import tensorflow as tf


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
