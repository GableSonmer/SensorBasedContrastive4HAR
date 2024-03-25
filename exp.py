import os
import sys

import h5py
import yaml

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from tensorflow.keras.backend import clear_session
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from keras.src.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.metrics import *
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers, models
import time
from tqdm import tqdm
from Augment import *
from SimCLRHAR import attach_projection_head, train_step, evaluate

transformations = [
    resampling_fast_random,
    resampling_random,
    noise,
    scaling,
    magnify,
    # inverting,
    reversing,
    time_warp
]
transformations_text = [
    'resampling_fast_random',
    'resampling_random',
    'noise',
    'scaling',
    'magnify',
    'inverting',
    'reversing',
    'time_warp'
]


def get_DCL(n_timesteps, n_features):
    input1 = Input((n_timesteps, n_features))
    # h1 = Conv1D(filters=128, kernel_size=5, activation='relu', name='conv1_1')(input1)
    # h2 = Conv1D(filters=128, kernel_size=5, activation='relu', name='conv1_2')(h1)
    # h3 = Conv1D(filters=128, kernel_size=5, activation='relu', name='conv1_3')(h2)
    # h4 = Conv1D(filters=128, kernel_size=5, activation='relu', name='conv1_4')(h3)
    # h = LSTM(128, return_sequences=True)(h4)
    # h = LSTM(128, return_sequences=False)(h)
    # return Model(input1, h)

    h1 = Conv1D(filters=32, kernel_size=5, activation='relu', name='conv1_1')(input1)
    h1 = BatchNormalization()(h1)
    h1 = MaxPooling1D(pool_size=2)(h1)

    h2 = Conv1D(filters=64, kernel_size=5, activation='relu', name='conv1_2')(h1)
    h2 = BatchNormalization()(h2)
    h2 = MaxPooling1D(pool_size=2)(h2)

    h3 = Conv1D(filters=128, kernel_size=5, activation='relu', name='conv1_3')(h2)
    h3 = BatchNormalization()(h3)
    h3 = MaxPooling1D(pool_size=2)(h3)

    h4 = Conv1D(filters=256, kernel_size=5, activation='relu', name='conv1_4')(h3)
    h4 = BatchNormalization()(h4)
    h4 = MaxPooling1D(pool_size=2)(h4)
    h = Flatten()(h4)
    return Model(input1, h)


class Exp:
    def __init__(self, method='simclr'):
        # load run.yml as config
        self.y_data = None
        self.x_data = None
        self.method = method
        with open(f'{method}.yml', 'r') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        # load dataset
        self.load_data()

    def run_supervised(self, iter):
        model = models.Sequential()
        model.add(layers.Conv1D(filters=32, kernel_size=5, activation='relu', input_shape=(250, 90)))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling1D(pool_size=2))

        model.add(layers.Conv1D(filters=64, kernel_size=5, activation='relu'))
        model.add(layers.MaxPooling1D(pool_size=2))
        model.add(layers.BatchNormalization())

        model.add(layers.Conv1D(filters=128, kernel_size=5, activation='relu'))
        model.add(layers.MaxPooling1D(pool_size=2))
        model.add(layers.BatchNormalization())

        model.add(layers.Conv1D(filters=256, kernel_size=5, activation='relu'))
        model.add(layers.MaxPooling1D(pool_size=2))
        model.add(layers.BatchNormalization())

        model.add(layers.Flatten())
        model.add(layers.Dense(100, activation='relu'))
        model.add(layers.Dense(7, activation='softmax'))

        model.compile(
            loss="sparse_categorical_crossentropy",
            metrics=['accuracy'],
            optimizer=tf.keras.optimizers.Adam(0.0001),

        )
        x_train, x_test, y_train, y_test = train_test_split(self.x_data, self.y_data, train_size=0.80, random_state=42)
        print('X train shape:', x_train.shape)
        print('y train shape:', y_train.shape)
        print('X test shape:', x_test.shape)
        print('y test shape:', y_test.shape)
        history = model.fit(
            x_train, y_train,
            epochs=100,
            batch_size=128,
            validation_data=(x_test, y_test),
            shuffle=True,
            callbacks=[CSVLogger('log_supervised.csv')],
            # verbose=0
        )
        best_acc = np.max(history.history['val_accuracy'])
        print(f'best accuracy: {best_acc * 100:.2f}')
        return best_acc

    def run(self, t1=None, t2=None):
        # run specified method
        if self.method == 'simclr':
            self.run_simclr(t1, t2)
        elif self.method == 'moco':
            self.run_moco(t1, t2)
        else:
            raise ValueError(f'Invalid method: {self.method}')

    def load_data(self):
        path = os.path.join('datasets', '{}.h5'.format(self.config['data_file']))
        with h5py.File(path, 'r') as hf:
            data = hf['data'][:]
            label = hf['label'][:]
        if self.config['data_file'] == 'mmfi':
            data = data.reshape(data.shape[0], 114, 30)
        else:
            data = data.reshape(data.shape[0], data.shape[1], -1)

        print(f'Loading {path}:')
        print('X shape:', data.shape)
        print('y shape:', label.shape)
        self.x_data = data
        self.y_data = label

    def run_simclr(self, t1=None, t2=None, fine_ratio=0.1):
        # select transformation
        if t1 is None or t2 is None:
            transform1 = transformations[transformations_text.index(self.config['transform1'])]
            transform2 = transformations[transformations_text.index(self.config['transform2'])]
        else:
            transform1 = t1
            transform2 = t2

        # IMPORTANT: clear the session before training
        clear_session()

        x_data = self.x_data
        y_data = self.y_data

        n_timesteps, n_features, n_outputs = x_data.shape[1], x_data.shape[2], self.config['num_classes']
        backbone = get_DCL(n_timesteps, n_features)
        model_cl = attach_projection_head(backbone)

        batch_size = self.config['batch_size']
        epochs = self.config['epochs']
        temperature = self.config['temperature']
        optimizer = Adam(self.config['learning_rate'])

        train_data, test_data, train_label, test_label = train_test_split(x_data, y_data,
                                                                          train_size=0.9,
                                                                          random_state=42)
        # split train into unlabeled and labeled, unlabeled for contrastive learning, labeled for fine-tuning
        # unlabled : labeled = 99:1
        unlabeled_data, finetune_data, _, finetune_label = train_test_split(train_data, train_label,
                                                                            train_size=1 - fine_ratio,
                                                                            random_state=42)
        print('Data Split')
        print('Test data shape:', test_data.shape, 'Test label shape:', test_label.shape)
        print('Unlabelled data shape:', train_data.shape)
        print('Finetune data shape:', finetune_data.shape, 'Finetune label shape:', finetune_label.shape)

        # early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

        # contrastive learning
        for epoch in tqdm(range(epochs)):
            loss_epoch = []
            train_loss_dataset = tf.data.Dataset.from_tensor_slices(train_data).shuffle(len(train_data),
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

        # classification data
        # train_x, test_x, train_y, test_y = train_test_split(labeled_data, labeled_label, train_size=0.90,
        #                                                     random_state=42)
        # linear evaluation
        linear_model = evaluate(model_cl, -6, n_outputs, 'linear')
        linear_model.compile(
            loss="sparse_categorical_crossentropy",
            metrics=['accuracy'],
            optimizer=tf.keras.optimizers.Adam(0.001)
        )

        # fit the model
        logger_path = os.path.join(self.config['save_path'], 'simclr', self.config['data_file'], 'logger')
        if not os.path.exists(logger_path):
            os.makedirs(logger_path)
        history_linear = linear_model.fit(
            finetune_data, finetune_label,
            epochs=100,
            batch_size=50,
            validation_data=(test_data, test_label),
            shuffle=True,
            verbose=0,
            callbacks=[
                # early_stopping,
                CSVLogger(
                    f'{logger_path}/log_linear_{transform1.__name__}_{transform2.__name__}.csv')]
        )

        linear_best_acc = np.max(history_linear.history['val_accuracy'])
        print(f'linear best accuracy: {linear_best_acc * 100:.2f}')

        # fine-tuning
        fine_model = evaluate(model_cl, -6, n_outputs, 'fine')
        fine_model.compile(
            loss="sparse_categorical_crossentropy",
            metrics=['accuracy'],
            optimizer=tf.keras.optimizers.Adam(0.001)
        )
        history_fine = fine_model.fit(
            finetune_data, finetune_label,
            epochs=100,
            batch_size=50,
            validation_data=(test_data, test_label),
            shuffle=True,
            verbose=0,
            callbacks=[
                # early_stopping,
                CSVLogger(
                    f'{logger_path}/log_finetune_{transform1.__name__}_{transform2.__name__}.csv')]
        )

        fine_best_acc = np.max(history_fine.history['val_accuracy'])
        print(f'fine best accuracy: {fine_best_acc * 100:.2f}')

        return linear_best_acc, fine_best_acc

    def run_moco(self, t1=None, t2=None):
        pass


exp = Exp()
i = 0.1
while i <= 0.5:
    print('Current fine ratio:', i)
    acc1_lst = []
    acc2_lst = []
    for _ in range(1):
        acc1, acc2 = exp.run_simclr(fine_ratio=i)
        acc1_lst.append(acc1)
        acc2_lst.append(acc2)
    print('Linear min {:.2f}, max {:.2f}, mean {:.2f}, std {:.2f}'.format(np.min(acc1_lst), np.max(acc1_lst),
                                                                          np.mean(acc1_lst), np.std(acc1_lst)))
    print('Fine min {:.2f}, max {:.2f}, mean {:.2f}, std {:.2f}'.format(np.min(acc2_lst), np.max(acc2_lst),
                                                                        np.mean(acc2_lst), np.std(acc2_lst)))
    print('=' * 40)
    i += 0.1

# n = len(transformations)
# acc_grid_linear = np.zeros((n, n))
# acc_grid_fine = np.zeros((n, n))
# for i1 in range(n):
#     for i2 in range(n):
#         t1, t2 = transformations[i1], transformations[i2]
#         print(f'Running simclr with {t1.__name__} and {t2.__name__}')
#         linear, fine = exp.run_simclr(t1, t2)
#         acc_grid_linear[i1, i2] = linear
#         acc_grid_fine[i1, i2] = fine
#         print('Done')
#
# # save to csv, with no header and index
# path = os.path.join(exp.config['save_path'], 'simclr', exp.config['data_file'], 'acc_records')
# if not os.path.exists(path):
#     os.makedirs(path)
# np.savetxt(f'{path}/acc_grid_linear.csv', acc_grid_linear, delimiter=',')
# np.savetxt(f'{path}/acc_grid_fine.csv', acc_grid_fine, delimiter=',')
# print('Saved to acc_grid_linear.csv and acc_grid_fine.csv')
