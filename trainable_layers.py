from tensorflow.keras.datasets import mnist
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

def define_dense_model_with_hidden_layers(input_length, 
                                          activation_func_array=['sigmoid', 'sigmoid'],
                                          hidden_layers_sizes=[50, 20],
                                          output_function='softmax',
                                          output_length=10):
    """Define a dense model with hidden layers."""
    model = keras.Sequential()

    # Add the first hidden layer with the input shape
    model.add(layers.Dense(hidden_layers_sizes[0], activation=activation_func_array[0], input_shape=(input_length,)))

    # Add additional hidden layers
    for size, activation in zip(hidden_layers_sizes[1:], activation_func_array[1:]):
        model.add(layers.Dense(size, activation=activation))

    # Add the output layer
    model.add(layers.Dense(output_length, activation=output_function))

    return model


def set_layers_to_trainable(model, trainable_layer_numbers):
    """Set specific layers of the model to be trainable or not."""
    for i, layer in enumerate(model.layers):
        if i in trainable_layer_numbers:
            layer.trainable = True
        else:
            layer.trainable = False
    return model