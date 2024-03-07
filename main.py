import os
import sys
import warnings

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
from keras.utils import to_categorical
from Augment import *
from SimCLRHAR import main as train_simclr
from MoCoHAR import main as train_moco

# set visible GPU
gpu_id = 0
gpus = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(gpus[gpu_id], 'GPU')
transformations = [
    resampling_fast_random,
    resampling_random,
    noise,
    scaling,
    magnify,
    inverting,
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

if __name__ == '__main__':
    config = 0  # 0 custom transformation, 1 all transformation
    files = ['self-collected.h5', 'mmfi.h5', 'that.h5']
    idx = 0

    # load data
    path = os.path.join('datasets', files[idx])
    with h5py.File(path, 'r') as hf:
        data = hf['data'][:]
        label = hf['label'][:]

    data = data.reshape(data.shape[0], data.shape[1], -1)
    # y need to convert to one-hot
    label = to_categorical(label)

    print(f'Loading {path}:')
    print('X shape:', data.shape)
    print('y shape:', label.shape)

    # Run Config 1: custom transformation
    if config == 0:
        i, j = 2, 5
        trans1 = transformations[i]
        trans2 = transformations[j]
        acc1, acc2 = train_moco(
            x_data=data, y_data=label,
            transform1=trans1,
            transform2=trans2
        )
        print(
            f'Pair {transformations_text[i]} & {transformations_text[j]}: Linear: {acc1:.2f} | Fine-tune: {acc2:.2f}')
    # Run Config 2: all transformation
    else:
        # grid accuracy of different 1st transformation and 2nd transformation
        n = len(transformations)
        grid_linear = np.zeros((n, n))
        grid_fine = np.zeros((n, n))

        ####################### Grid Fine-tune Evaluation #######################
        for i in range(n):
            for j in [4, 5, 6]:
                trans1 = transformations[i]
                trans2 = transformations[j]
                acc1, acc2 = train_simclr(
                    x_data=data, y_data=label,
                    transform1=trans1,
                    transform2=trans2
                )
                grid_linear[i, j] = round(acc1, 2)
                grid_fine[i, j] = round(acc2, 2)
                print(
                    f'Pair {transformations_text[i]} & {transformations_text[j]}: Linear: {acc1:.2f} | Fine-tune: {acc2:.2f}')
        ####################### Grid Fine-tune Evaluation #######################

        # save grid learner to csv
        np.savetxt('linear.csv', grid_linear, fmt='%.4f', delimiter=',')
        np.savetxt('fine.csv', grid_fine, fmt='%.4f', delimiter=',')

        # use seaborn to plot the grid accuracy
        plt.figure(figsize=(30, 15), dpi=100)

        # 第一个子图
        plt.subplot(1, 2, 1)
        sns.heatmap(grid_linear, annot=True, cmap='YlGnBu', cbar=False,
                    xticklabels=transformations_text, yticklabels=transformations_text,
                    annot_kws={"size": 25})  # 调整注释文字大小
        plt.title('Linear Evaluation', fontsize=30)  # 调整标题大小
        plt.xticks(rotation=90, fontsize=30)
        plt.yticks(fontsize=30)

        # 第二个子图
        plt.subplot(1, 2, 2)
        sns.heatmap(grid_fine, annot=True, cmap='YlGnBu', cbar=False, xticklabels=transformations_text,
                    annot_kws={"size": 25})  # 同样调整注释文字大小
        plt.title('Fine-tune Evaluation', fontsize=30)  # 调整标题大小
        plt.xticks(rotation=90, fontsize=30)
        plt.yticks(fontsize=30)

        plt.subplots_adjust(left=0.25, bottom=0.4, right=0.99, top=0.95)

        plt.savefig('contrastive_model/simclr/acc_heatmap.png')
        plt.show()
