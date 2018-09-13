import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt 
from keras.layers import Input, Dense, Conv1D, Dropout, MaxPooling1D, Flatten
from keras.models import Model
from keras import optimizers
import pdb

num_markers = 5000

inp = Input(shape=(num_markers,1))
x = Conv1D(8,18,activation='relu')(inp)
x = MaxPooling1D(pool_size=4,stride=4)(x)
x = Dropout(0.2)(x)
x = Flatten()(x)
x = Dense(32,activation='relu')(x)
x = Dropout(0.1)(x)
x = Dense(1)(x)
out = Dropout(0.05)(x)

model = Model(inputs=inp, outputs=out)
sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.5, nesterov=False)
model.compile(optimizer='sgd',
              loss='mean_absolute_error',
              metrics=['mae'])



x_data = pd.read_pickle('x.pkl')
y_data = pd.read_pickle('y.pkl')

train = x_data.as_matrix()[:,:num_markers]
labels = y_data.as_matrix()[:,0]

h = model.fit(train[...,None], labels, batch_size=16, validation_split=0.1, epochs=500)  # starts training

plt.figure()
plt.plot(h.history['loss'])
plt.figure()
plt.plot(h.history['val_loss'])
plt.show()