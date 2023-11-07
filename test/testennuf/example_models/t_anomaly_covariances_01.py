#  (C) Crown Copyright, Met Office, 2023.
import tensorflow as tf


class TAnomalyCovariances01:
    """This model is meant to take 6 simple inputs and output the lag<2 part of the covariance matrix
    of temperature anomaly"""

    @staticmethod
    def build_function(alpha=0.1) -> tf.keras.Model:
        """Functional API model, entirely dense"""
        # Input reshaping
        inputs = tf.keras.Input(shape=6, name="inputs")
        activation = tf.keras.layers.LeakyReLU(alpha=alpha)
        nnodes = 256
        reg = None
        y = tf.keras.layers.Dense(nnodes, activation=activation, kernel_regularizer=reg, name="dense1")(inputs)
        y = tf.keras.layers.Dense(nnodes, activation=activation, kernel_regularizer=reg, name="dense2")(y)
        y = tf.keras.layers.Dense(nnodes, activation=activation, kernel_regularizer=reg, name="dense3")(y)
        y = tf.keras.layers.Dense(nnodes, activation=activation, kernel_regularizer=reg, name="dense4")(y)

        z = tf.keras.layers.Dense(nnodes, activation=activation, kernel_regularizer=reg, name="covdense1")(inputs)
        z = tf.keras.layers.Dense(nnodes, activation=activation, kernel_regularizer=reg, name="covdense2")(z)
        z = tf.keras.layers.Dense(nnodes, activation=activation, kernel_regularizer=reg, name="covdense3")(z)
        z = tf.keras.layers.Dense(nnodes, activation=activation, kernel_regularizer=reg, name="covdense4")(z)
        # y = tf.keras.layers.Dense(nnodes, activation=activation, kernel_regularizer=reg, name='dense5')(y)
        # y = tf.keras.layers.Dense(nnodes, activation=activation, name='dense6')(y)
        # y = tf.keras.layers.Dense(nnodes, activation=activation, name='dense7')(y)
        # y = tf.keras.layers.Dense(nnodes, activation=activation, name='dense8')(y)
        stddev = tf.keras.layers.Dense(70, activation="relu", name="stddevs")(y)
        covariances = tf.keras.layers.Dense(69 + 68, name="covariances")(z)
        outputs = tf.keras.layers.Concatenate(name="outputs")([stddev, covariances])
        return tf.keras.Model(inputs=inputs, outputs=outputs)
