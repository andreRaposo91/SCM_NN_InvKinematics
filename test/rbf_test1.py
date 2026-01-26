import numpy as np
import keras
from keras import Model, Sequential
from keras.layers import Layer, Dense
from keras import backend as K

class RBFLayer(Layer):
    def __init__(self, units, gamma, **kwargs):
        super(RBFLayer, self).__init__(**kwargs)
        self.units = units
        self.gamma = K.cast_to_floatx(gamma)

    def build(self, input_shape):
        self.mu = self.add_weight(name='mu',
                                  shape=(int(input_shape[1]), self.units),
                                  initializer='uniform',
                                  trainable=True)
        super(RBFLayer, self).build(input_shape)

    def call(self, inputs):
        diff = K.expand_dims(inputs) - self.mu
        l2 = K.sum(K.pow(diff,2), axis=1)
        res = K.exp(-1 * self.gamma * l2)
        return res

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)


# def RBFNN(data, optimizer='adam', loss='mse'):
#     input_shape = (data.shape[1],)
#     input_layer = keras.layers.Input(shape=input_shape)
#     # model.add(Dense(20, activation='relu', input_shape=input_shape))
#     rbf_layer = RBFLayer(20, 0.5)(input_layer)
#     output_layer = keras.layers.Dense(3)(rbf_layer)
#     model = Model(inputs=input_layer, outputs=output_layer)
#     model.compile(optimizer=optimizer, loss=loss)
    # return model

def RBFNN(data, optimizer='adam', loss='mse'):
    input_shape = (data.shape[1],)
    input_layer = keras.layers.Input(shape=input_shape)
    model = Sequential()
    # model.add(Dense(20, activation='relu', input_shape=input_shape))
    model.add(RBFLayer(20, 0.5, input_shape=input_shape))
    model.add(Dense(1))
    model.compile(optimizer=optimizer, loss=loss)
    return model

x_train = np.random.rand(100, 2)  
y_train = np.sin(2 * np.pi * x_train[:, 0]) + np.cos(2 * np.pi * x_train[:, 1])  # Target values

rbf_model = RBFNN(x_train)
rbf_model.fit(x_train, y_train, epochs=50, batch_size=20, validation_split=0.2)
print(rbf_model.evaluate(x_train, y_train))


