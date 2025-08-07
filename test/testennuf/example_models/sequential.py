import tensorflow as tf


class SequentialExamples:
    @staticmethod
    def simple_mlp():
        model = tf.keras.Sequential()
        model.add(tf.keras.Input(shape=(4,)))
        model.add(tf.keras.layers.Dense(8))
        model.add(tf.keras.layers.Dense(8))
        model.add(tf.keras.layers.Dense(2))
        return model

    @staticmethod
    def simple_mlp_with_activations_as_layers():
        model = tf.keras.Sequential()
        model.add(tf.keras.Input(shape=(4,)))
        model.add(tf.keras.layers.Dense(8))
        model.add(tf.keras.layers.ReLU())
        model.add(tf.keras.layers.Dense(8))
        model.add(tf.keras.layers.ReLU())
        model.add(tf.keras.layers.Dense(2))
        return model

    @staticmethod
    def simple_mlp_with_activations():
        model = tf.keras.Sequential()
        model.add(tf.keras.Input(shape=(4,)))
        model.add(tf.keras.layers.Dense(8, activation="tanh"))
        model.add(tf.keras.layers.Dense(8, activation="relu"))
        model.add(tf.keras.layers.Dense(2))
        return model
