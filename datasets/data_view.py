import numpy
import numpy as np
import h5py

x = np.load('UCI_X.npy')
y = np.load('UCI_Y.npy')

# check the shape of the data
print(x.shape)
print(y.shape)
print(y[0])

with h5py.File('self-collected.h5', 'r') as hf:
    data = hf['data'][:]
    label = hf['label'][:]

print(data.shape)
print(label.shape)

