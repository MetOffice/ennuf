#  (C) Crown Copyright, Met Office, 2023.
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
        y = tf.keras.layers.Dense(nnodes, activation=activation, kernel_regularizer=reg, name='dense1')(scalars)
        y = tf.keras.layers.Dense(nnodes, activation='sigmoid', kernel_regularizer=reg, name='dense2')(y)
        y = tf.keras.layers.Dense(nnodes, activation='relu', kernel_regularizer=reg, name='dense3')(y)
        y = tf.keras.layers.Dense(nnodes, activation='tanh', kernel_regularizer=reg, name='dense4')(y)
        outputs_1 = tf.keras.layers.Dense(1, activation='relu', name='outputs_1')(y)
        outputs_2 = tf.keras.layers.Dense(2, activation='relu', name='outputs_2')(y)
        return tf.keras.models.Model(inputs=scalars, outputs={'outputs_1': outputs_1, 'outputs_2': outputs_2})

    @staticmethod
    def build_sequential():
        nnodes = 8
        model = tf.keras.Sequential()
        model.add(tf.keras.Input(6))
        for activation in ['relu', 'sigmoid', 'tanh']:
            model.add(tf.keras.layers.Dense(nnodes, activation=activation))
        return model.compile()
