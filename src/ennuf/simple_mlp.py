#  (C) Crown Copyright, Met Office, 2023.
from keras.layers import LeakyReLU
# from tensorflow.python.keras import Model, Input, layers, models
import tensorflow as tf


class SimpleMLP:
    @staticmethod
    def build():
        """Builds some machine learning model"""
        # FLAG
        alpha = 0.1
        nnodes = 8
        activation = 'relu'
        reg = None
        scalars = tf.keras.Input(shape=6, name='scalars')
        y = tf.keras.layers.Dense(nnodes, activation=activation, kernel_regularizer=reg, name='dense1', use_bias=False)(scalars)
        y = tf.keras.layers.Dense(nnodes, activation='sigmoid', kernel_regularizer=reg, name='dense2')(y)
        y = tf.keras.layers.Dense(nnodes, activation=None, kernel_regularizer=reg, name='dense3')(y)
        y = tf.keras.layers.Dense(nnodes, activation=LeakyReLU(alpha), kernel_regularizer=reg, name='dense4')(y)
        outputs = tf.keras.layers.Dense(1, activation=LeakyReLU(alpha), name='outputs')(y)
        return tf.keras.models.Model(inputs=scalars, outputs=outputs)
