from typing_extensions import override
import numpy as np
import matplotlib.pyplot as plt
# TensorFlow and Keras libraries for deep learning
# import tensorflow.compat.v1 as tf
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from keras.models import Sequential

# Other imports (if necessary)
from keras.layers import SimpleRNN as SimpleRNN_layer, LSTM as LSTM_layer
from keras.layers import Layer
from keras import backend as K
from keras.callbacks import Callback, EarlyStopping


class DNNModelBuilder:
    def __init__(self, input_data, architectures):
        """
        :param input_data: data used to determine the input shape
        :param architectures: list of functions. Each function should return a compiled tf.keras.model.
        """
        self.input_data = input_data
        self.architectures = architectures
        self.models = [arch(input_data, *args, **kwargs) for arch, args, kwargs in architectures]
        self.histories = []

    def train(self, data, labels, epochs=50, batch_size=32, validation_split=0.2, patience=100, validation_data=([], [])):
        self.early_stopping = None
        for i, model in enumerate(self.models):
            print(f"Training model {i + 1}/{len(self.models)}")
            if patience == -1:
                if validation_data == ([], []):
                    history = model.fit(data, labels, epochs=epochs, batch_size=batch_size, validation_split=validation_split, verbose=0)
                else:
                    history = model.fit(data, labels, epochs=epochs, batch_size=batch_size, validation_data=validation_data, verbose=0)
            else:
                # self.early_stopping = EarlyStopping(monitor='val_loss', patience=patience, verbose=1, mode='min', restore_best_weights=True)
                self.early_stopping = MovingAverageEarlyStopping(monitor='val_loss', patience=patience, min_delta=0.01, min_loss=25)
                if validation_data == ([], []):
                    history = model.fit(data, labels, epochs=epochs, batch_size=batch_size, validation_split=validation_split, verbose=0, callbacks=[self.early_stopping])
                else:
                    history = model.fit(data, labels, epochs=epochs, batch_size=batch_size, validation_data=validation_data, verbose=0, callbacks=[self.early_stopping])
            self.histories.append(history)  # Store training history for future analysis

    def predict(self, data):
        predictions = []
        for model in self.models:
            predictions.append(model.predict(data))
        return predictions

    def evaluate(self, data, labels):
        """Evaluate models on given data and labels, returns loss values."""
        losses = []
        for model in self.models:
            loss = model.evaluate(data, labels)
            losses.append(loss)
        return losses

    def save_models(self, prefix="model", folder='.'):
        """Store models to disk."""
        # if len(self.architectures[0]) == 1:
        #     for i, model in enumerate(self.models):
        #         model.save(f"{prefix}_{i}.keras")
        # else:
        #     for i, model in enumerate(self.models):
        #         model.save(f"{prefix}_{i}_{self.architectures[i][1]}.keras")
        for i, model in enumerate(self.models):
            layer_size = str(self.architectures[i][1]).strip("[]").strip("()").split(',')[0]
            name = f"{folder}/{prefix}_{layer_size}.keras"
            model.save(name)

    def compare_models(self, models=[], stack=True):
        """Compare training histories visually."""
        if not stack: plt.figure()
        if models == []:
            models = len(self.histories)
        for i, history in enumerate(self.histories[:models]):
            if self.architectures[i][1]:
                plt.plot(history.history['val_loss'],
                    label=f"Model {i + 1}: {str(self.architectures[i][0]).split(' ')[1]}, {str(self.architectures[i][1][0])}")
            else: plt.plot(history.history['val_loss'],
                    label=f"Model {i + 1}: {str(self.architectures[i][0]).split(' ')[1]}")

        plt.title('Model Comparison on Validation Loss')
        plt.ylabel('Validation Loss')
        plt.xlabel('Epoch')
        plt.legend()
        plt.ylim([0, np.mean([np.mean(history.history['val_loss']) for history in self.histories]).astype(float) * 1.5])
        # plt.show()


##### Several types of architectures

def mlp_1(data, layers, activation='relu', optimizer='adam', loss='mse'):
    input_shape = (data.shape[1],)
    model = Sequential()
    model.add(Dense(layers[0], activation=activation, input_shape=input_shape))
    if len(layers) > 1:
        for layer in layers[1:]:
            model.add(Dense(layer, activation=activation))
    model.add(Dense(3))
    model.compile(optimizer=optimizer, loss=loss)
    return model

def rnn(data, layers, optimizer='adam', loss='mse', activation='relu'):
    """
    Recurrent Neural Network (RNN)
    Useful for temporal sequences.

    layers: layers: list[args: tuple[units: int, ret_seq: bool]]
    """
    # Check for 3D shape, if not, try to reshape
    if len(data.shape) == 2:
        data = data.reshape(1, data.shape[0], data.shape[1])
    # input_shape = (None, data.shape[2])
    # input_shape = (None, data.shape[1], data.shape[2])
    input_shape = (data.shape[1], data.shape[2])
    # input_shape = (data.shape[0], data.shape[2])
    model = Sequential()
    print(data.shape, '|', input_shape, '|', *layers)
    ret_seq_flag = True
    if len(layers) >= 2:
        # model.add(SimpleRNN_layer(layers[0][0], return_sequences=True, input_shape=input_shape, activation=activation))
        for units, ret_seq in layers[0:-1]:
            if ret_seq_flag and ret_seq:
                model.add(SimpleRNN_layer(units, return_sequences=True, activation=activation))
            elif ret_seq_flag and not ret_seq:
                model.add(SimpleRNN_layer(units, return_sequences=False, activation=activation))
                ret_seq_flag = False
            else: 
                model.add(Dense(units, activation=activation))
        if ret_seq_flag:
            model.add(SimpleRNN_layer(layers[-1][0], return_sequences=False, activation=activation))
        else:
            model.add(Dense(layers[-1][0], activation=activation))
        # for units, ret_seq in layers[1:-1]:
        # if layers[-2][1]:
        #     model.add(SimpleRNN_layer(layers[-1][0], return_sequences=True, activation=activation))
        # else: model.add(SimpleRNN_layer(layers[-1][0], return_sequences=False,activation=activation))
    elif len(layers) == 1:
        model.add(SimpleRNN_layer(layers[-1][0], return_sequences=False, activation=activation))
    model.add(Dense(3))
    model.compile(optimizer=optimizer, loss=loss)
    model.build(data.shape)
    # print(model.summary())
    return model

def lstm(data, layers, optimizer='adam', loss='mse'):
    # Check for 3D shape, if not, try to reshape
    if len(data.shape) == 2:
        data = data.reshape(data.shape[0], data.shape[1], 1)
    input_shape = (data.shape[0], data.shape[2])
    model = Sequential()
    if len(layers) >= 2: 
        model.add(LSTM_layer(layers[0][0], return_sequences=layers[0][1], input_shape=input_shape))
        for units, ret_seq in layers[1:-1]:
            model.add(LSTM_layer(units, return_sequences=ret_seq))
        if layers[-2][1]:
            model.add(LSTM_layer(layers[-1][0], return_sequences=False))
        else: model.add(Dense(layers[-1][0]))
    elif len(layers) == 1:
        model.add(LSTM_layer(layers[-1][0], return_sequences=False))
    model.add(Dense(3))
    model.compile(optimizer=optimizer, loss=loss)
    model.build(data.shape)
    print(model.summary())
    print(data.shape, '|', input_shape, '|', layers)
    return model

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

def SimpleRBFNN(data, optimizer='adam', loss='mse'):
    input_shape = (data.shape[1],)
    # input_layer = keras.layers.Input(shape=input_shape)
    model = Sequential()
    # model.add(Dense(20, activation='relu', input_shape=input_shape))
    model.add(RBFLayer(20, 0.5, input_shape=input_shape))
    model.add(Dense(3))
    model.compile(optimizer=optimizer, loss=loss)
    return model


def SimpleRNN(data, optimizer='adam', loss='mse', activation='relu'):
    """
    Recurrent Neural Network (RNN)
    Useful for temporal sequences.
    """
    # Check for 3D shape, if not, try to reshape
    if len(data.shape) == 2:
        data = data.reshape(1, data.shape[0], data.shape[1])
    input_shape = (None, data.shape[2])
    input_shape = (data.shape[1], data.shape[2])
    print(data.shape, '\n', input_shape)
    model = Sequential()
    model.add(SimpleRNN_layer(20, return_sequences=True, input_shape=input_shape, activation=activation))
    # model.add(SimpleRNN_layer(20, return_sequences=True, activation=activation))
    # model.add(SimpleRNN_layer(20))
    model.add(Dense(20))
    model.add(Dense(3))
    model.build(data.shape)
    print(model.summary())
    model.compile(optimizer=optimizer, loss=loss)
    return model

def LSTM_NN(data, optimizer='adam', loss='mse'):
    if len(data.shape) == 2:
        data = data.reshape(data.shape[0], data.shape[1], 1)
    input_shape = (data.shape[1], data.shape[2])
    # input_shape = data.shape
    model = Sequential()
    # input_layer = keras.layers.Input(shape=input)
    # model.add(Input(shape=input))
    model.add(LSTM_layer(30, return_sequences=False, input_shape=input_shape))
    model.add(LSTM_layer(30, return_sequences=False))
    model.add(Dense(3))
    model.compile(optimizer=optimizer, loss=loss)
    return model

def ResNet(data, optimizer='adam', loss='mse'):
    """
    Deep Residual Network (ResNet)
    A deep network with skip connections.
    
    # Example Usage:
    model_resnet = ResNet(processed_data)
    model_resnet.fit(processed_data, target_positions, epochs=10)
    """
    input_shape = (data.shape[1],)
    input_layer = tf.keras.layers.Input(shape=input_shape)
    x = Dense(128, activation='relu')(input_layer)
    for _ in range(3):
        y = Dense(128, activation='relu')(x)
        y = Dense(128, activation='relu')(y)
        x = tf.keras.layers.add([x, y])
    x = Dense(64, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    output_layer = Dense(3)(x)
    model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=optimizer, loss=loss)
    return model

def DropoutDNN(data, optimizer='adam', loss='mse'):
    """
    Dropout Regularized (DNN)
    A feedforward architecture with dropout layers.
    
    # Example Usage:
    model_dropout = DropoutDNN(processed_data)
    model_dropout.fit(processed_data, target_positions, epochs=10)
    """
    input_shape = (data.shape[1],)
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=input_shape))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(3))
    model.compile(optimizer=optimizer, loss=loss)
    return model


class MovingAverageEarlyStopping(Callback):
    def __init__(self, monitor="val_loss", patience=100, window_size=20, min_delta=0.01, min_loss=10):
        super(MovingAverageEarlyStopping, self).__init__()
        self.patience = patience
        self.window_size = window_size
        self.min_delta = min_delta
        self.wait = 0
        self.stopped_epoch = 0
        self.val_losses = []
        self.best_loss = float('inf')
        self.monitor = monitor
        self.min_loss = min_loss
        self.best_weights = None

    def on_epoch_end(self, epoch, logs):
        current_val_loss = logs.get(self.monitor)
        self.val_losses.append(current_val_loss)
        
        if len(self.val_losses) > self.window_size:
            moving_average_loss = np.mean(self.val_losses[-self.window_size:])
            if moving_average_loss < self.best_loss - self.min_delta:
                self.best_loss = moving_average_loss
                self.wait = 0
                self.best_weights = self.model.get_weights()  # Save the current best weights
            else:
                self.wait += 1
                if self.wait >= self.patience and self.best_loss <= self.min_loss:
                    self.stopped_epoch = epoch
                    self.model.stop_training = True
                    print(f"Moving Average early stopping: Training stopped at epoch {self.stopped_epoch+1}, "
                          f"val_loss moving average didn't improve for {self.patience} consecutive epochs.")
                    # Restore the model to the state with the best loss
                    if self.best_weights is not None:
                        self.model.set_weights(self.best_weights)
