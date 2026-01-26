# import tensorflow as tf
import sys
import tensorflow.compat.v1 as tf 
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

config = tf.ConfigProto(device_count = {'GPU': 1})

import numpy as np
import matplotlib.pyplot as plt

# # x = np.linspace(0, np.pi*2, 200)
# # y_test = np.sin(x)

# x = np.linspace(0, 10, 200)
# y_test = (x**3 - x**2) * 0.1

# y_train = y_test + np.random.rand(len(x))*0.1


# model = Sequential()
# model.add(Dense(20, input_shape=(1,), activation='relu'))
# # model.add(Dense(50))
# model.add(Dense(1))

# optimizer = Adam(lr=0.01)
# model.compile(optimizer='adam', loss='mse')

# print(model.summary())

# sys.exit()
# # history = model.fit(x, y_train, epochs=200, validation_split=0.1, verbose=0)

# plt.plot(history.history['val_loss'])

# plt.figure()
# pred = model.predict(x)
# print(pred.shape)
# print(pred[:10])
# print(y_test[:10])
# plt.plot(x, pred, label='predict')
# plt.plot(x, y_train, label='train')
# plt.plot(x, y_test, label='test')
# plt.legend()

# plt.show()

# import tensorflow as tf
# from keras.models import Sequential
# from keras.layers import Dense, Activation
# from keras.optimizers import Adam
# import numpy as np
# import matplotlib.pyplot as plt

# Generate sample data
x = np.linspace(0, 10, 200)
y_test = (x**3 - x**2) * 0.1
y_train = y_test + np.random.rand(len(x)) * 0.1

# Build the model
model = Sequential([
    Dense(50, input_shape=(1,), activation='relu'),
    Dense(1)
])

# Compile the model
# optimizer = Adam(learning_rate=0.001)
optimizer = Adam(lr=0.001)
model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

print(model.summary())
sys.exit()
# # Train the model
# history = model.fit(x, y_train, epochs=200, validation_split=0.1, verbose=0)

# # Plot loss curve
# plt.plot(history.history['val_loss'])
# plt.xlabel('Epochs')
# plt.ylabel('Validation Loss')
# plt.title('Validation Loss Over Epochs')


# # Plot predictions
# plt.figure()
# pred = model.predict(x)
# plt.plot(x, pred, label='Predictions')
# plt.plot(x, y_train, label='Training Data')
# plt.plot(x, y_test, label='Test Data')
# plt.legend()
# plt.xlabel('Input')
# plt.ylabel('Output')
# plt.title('Model Predictions')
# plt.show()
