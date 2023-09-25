#  (C) Crown Copyright, Met Office, 2023.
import tensorflow as tf


class SimpleMLP:
    @staticmethod
    def build_functional():
        """Builds some machine learning model with the keras functional API"""
        nnodes = 8
        activation = 'relu'
        reg = None
        scalars = tf.keras.Input(shape=6, name='scalars')
        y = tf.keras.layers.Dense(nnodes, activation=activation, kernel_regularizer=reg, name='dense1')(scalars)
        y = tf.keras.layers.Dense(nnodes, activation='sigmoid', kernel_regularizer=reg, name='dense2')(y)
        y = tf.keras.layers.Dense(nnodes, activation='relu', kernel_regularizer=reg, name='dense3')(y)
        y = tf.keras.layers.Dense(nnodes, activation='tanh', kernel_regularizer=reg, name='dense4')(y)
        outputs_1 = tf.keras.layers.Dense(1, name='outputs_1')(y)
        outputs_2 = tf.keras.layers.Dense(2, name='outputs_2')(y)
        return tf.keras.models.Model(inputs=scalars, outputs={'outputs_1': outputs_1, 'outputs_2': outputs_2})

    @staticmethod
    def build_functional_1_1():
        """Builds some machine learning model with the keras functional API"""
        nnodes = 8
        activation = 'relu'
        reg = None
        scalars = tf.keras.Input(shape=6, name='scalars')
        y = tf.keras.layers.Dense(nnodes, activation=activation, kernel_regularizer=reg, name='dense1')(scalars)
        y = tf.keras.layers.Dense(nnodes, activation='sigmoid', kernel_regularizer=reg, name='dense2')(y)
        y = tf.keras.layers.Dense(nnodes, activation='relu', kernel_regularizer=reg, name='dense3')(y)
        y = tf.keras.layers.Dense(nnodes, activation='tanh', kernel_regularizer=reg, name='dense4')(y)
        outputs_1 = tf.keras.layers.Dense(1, name='outputs_1')(y)
        outputs_2 = tf.keras.layers.Dense(2, name='outputs_2')(y)
        return tf.keras.models.Model(inputs=scalars, outputs={'foo': outputs_1, 'bar': outputs_2})

    @staticmethod
    def build_functional_2():
        """Builds some machine learning model with the keras functional API"""
        nnodes = 32
        activation = 'relu'
        reg = None
        scalars = tf.keras.Input(shape=6, name='scalars')
        profile = tf.keras.Input(shape=70, name='profile')
        y = tf.keras.layers.Dense(nnodes, activation=activation, kernel_regularizer=reg, name='dense1')(scalars)
        y = tf.keras.layers.Dense(nnodes, activation='sigmoid', kernel_regularizer=reg, name='dense2')(y)
        y = tf.keras.layers.Dense(nnodes, activation='relu', kernel_regularizer=reg, name='dense3')(y)
        y = tf.keras.layers.Dense(nnodes, activation='tanh', kernel_regularizer=reg, name='dense4')(y)
        z = tf.keras.layers.Dense(nnodes, activation=activation, kernel_regularizer=reg, name='dense1')(profile)
        z = tf.keras.layers.Dense(nnodes, activation=None, kernel_regularizer=reg, name='dense2')(z)
        z = tf.keras.layers.Dense(nnodes, activation='relu', kernel_regularizer=reg, name='dense3')(z)
        z = tf.keras.layers.Dense(nnodes, activation='tanh', kernel_regularizer=reg, name='dense4')(z)
        outputs_1 = tf.keras.layers.Dense(9, name='outputs_1')(y)
        outputs_2 = tf.keras.layers.Dense(2, name='outputs_2')(y)
        outputs_3 = tf.keras.layers.Dense(3, name='outputs_3')(z)
        return tf.keras.models.Model(
            inputs={'scalars': scalars, 'profile': profile},
            outputs={'outputs_1': outputs_1, 'outputs_2': outputs_2, 'outputs_3': outputs_3}
            )

    @staticmethod
    def build_sequential():
        nnodes = 8
        model = tf.keras.Sequential()
        model.add(tf.keras.Input(6))
        for activation in ['relu', 'sigmoid', 'tanh']:
            model.add(tf.keras.layers.Dense(nnodes, activation=activation))
        return model.compile()
