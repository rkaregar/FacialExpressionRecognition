# %%
import numpy as np
import pandas as pd
from time import time
from matplotlib import pyplot as plt
# %%
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
# %%
import keras
from keras.models import Sequential, model_from_json
from keras.callbacks import TensorBoard
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras import backend as K
# %%
img_size = 48
img_rows, img_cols = 48, 48
batch_size = 32
num_classes = 7
epochs = 6
# %%
data = pd.read_csv('data.csv', header=0)
data2 = data['pixels'].str.split(expand=True)
cleaned_data = pd.concat([data[['emotion', 'Usage']], data2], axis=1)
cleaned_data.to_csv('cleaned_data.csv', index=False)
# %%
data = pd.read_csv('cleaned_data.csv', header=0)
# %%
data.iloc[:, 2:] = data.iloc[:, 2:] / 256.
# %%
train = data[data['Usage'] == 'Training'].drop('Usage', axis=1)
public_test = data[data['Usage'] == 'PublicTest'].drop('Usage', axis=1)
private_test = data[data['Usage'] == 'PrivateTest'].drop('Usage', axis=1)
# %%
x_train = train.values[:, 1:]
y_train = train.values[:, 0]
x_public_test = public_test.values[:, 1:]
y_public_test = public_test.values[:, 0]
x_private_test = private_test.values[:, 1:]
y_private_test = private_test.values[:, 0]
# %%
for i in range(1, 10):
    plt.subplot(3, 3, i)
    plt.imshow(x_train[100 + i * 10].reshape(img_size, -1), cmap='gray')
    plt.title('image: {}'.format(i))
plt.show()
# %%
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_public_test = x_public_test.reshape(x_public_test.shape[0], 1, img_rows, img_cols)
    x_private_test = x_private_test.reshape(x_private_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_public_test = x_public_test.reshape(x_public_test.shape[0], img_rows, img_cols, 1)
    x_private_test = x_private_test.reshape(x_private_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

y_train = keras.utils.to_categorical(y_train, num_classes)
y_public_test = keras.utils.to_categorical(y_public_test, num_classes)
y_private_test = keras.utils.to_categorical(y_private_test, num_classes)
# %%
model = Sequential()
model.add(Conv2D(256, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
# %%
tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_public_test, y_public_test))
score = model.evaluate(x_public_test, y_public_test, verbose=0)
# %%
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(x_train, y_train)
knn.score(x_public_test[:1000, :], y_public_test[:1000])
# %%
y_pred = knn.predict(x_public_test)
# %%
data.groupby('emotion').size() / data.shape[0]
# %%
np.divide(100 * confusion_matrix(y_public_test, y_pred),
          data[data['Usage'] == 'PublicTest'].groupby('emotion').size().values)
# %%
# expressions: anger, disgust, fear, happiness, sadness, surprise, and neutral

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_public_test = x_public_test.reshape(x_public_test.shape[0], 1, img_rows, img_cols)
    x_private_test = x_private_test.reshape(x_private_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_public_test = x_public_test.reshape(x_public_test.shape[0], img_rows, img_cols, 1)
    x_private_test = x_private_test.reshape(x_private_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

y_train = keras.utils.to_categorical(y_train, num_classes)
y_public_test = keras.utils.to_categorical(y_public_test, num_classes)
y_private_test = keras.utils.to_categorical(y_private_test, num_classes)
# %%
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights('model.h5')
# %%
loaded_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(),
                     metrics=['accuracy'])
score = loaded_model.evaluate(x_public_test, y_public_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1] * 100))
# %%
weight_conv2d_1 = loaded_model.layers[4].get_weights()[0][:, :, 0, :]

col_size = 6
row_size = 5
filter_index = 0
fig, ax = plt.subplots(row_size, col_size, figsize=(12, 8))
for row in range(0, row_size):
    for col in range(0, col_size):
        ax[row][col].imshow(weight_conv2d_1[:, :, filter_index], cmap="gray")
        filter_index += 1

plt.show()
