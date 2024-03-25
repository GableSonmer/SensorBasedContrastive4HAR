import numpy
import numpy as np
import h5py

# with h5py.File('mmfi.h5', 'r') as hf:
#     data = hf['data'][:]
#     label = hf['label'][:]
#
# print(data.shape)
# print(label.shape)
# data = data.reshape(data.shape[0], 114, 30)
# print(data.shape)
#
# x = np.load('UCI_X.npy')
# y = np.load('UCI_Y.npy')
# print(x.shape, y.shape)

X_train = np.load('./UT_HAR/data/X_train.npy')
X_test = np.load('./UT_HAR/data/X_test.npy')
X_val = np.load('./UT_HAR/data/X_val.npy')
y_train = np.load('./UT_HAR/label/y_train.npy')
y_test = np.load('./UT_HAR/label/y_test.npy')
y_val = np.load('./UT_HAR/label/y_val.npy')

print('Train:', X_train.shape, y_train.shape)
print('Test:', X_test.shape, y_test.shape)
print('Val:', X_val.shape, y_val.shape)

# contact into data and label
data = np.concatenate((X_train, X_test, X_val), axis=0)
label = np.concatenate((y_train, y_test, y_val), axis=0)

# save into h5
with h5py.File('ut_har.h5', 'w') as hf:
    hf.create_dataset('data', data=data)
    hf.create_dataset('label', data=label)
hf.close()
