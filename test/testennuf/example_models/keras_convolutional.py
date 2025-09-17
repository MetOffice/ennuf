# python
import tensorflow as tf

class KerasConvolutional:
    @staticmethod
    def only_flatten():
        inputs = tf.keras.Input(shape=(3,2,4), name="inputs")
        outputs = tf.keras.layers.Flatten()(inputs)
        return tf.keras.Model(inputs=inputs, outputs=outputs)

    @staticmethod
    def build_only_conv():
        inputs = tf.keras.Input(shape=(4, 2), name="inputs")
        x = tf.keras.layers.Conv1D(2, 3, activation=None, padding="same")(inputs)
        x = tf.keras.layers.Conv1D(2, 3, activation=None, padding="same")(x)
        x = tf.keras.layers.Flatten()(x)

        return tf.keras.Model(inputs=inputs, outputs=x)

    @staticmethod
    def build_conv_with_dense():
        inputs = tf.keras.Input(shape=(8, 2), name="inputs")
        x = tf.keras.layers.Conv1D(2, 3, activation="relu", padding="same")(inputs)
        x = tf.keras.layers.Dense(4)(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(4, activation="tanh")(x)
        x = tf.keras.layers.Dense(4)(x)
        return tf.keras.Model(inputs=inputs, outputs=x)

    @staticmethod
    def build_simple_conv():
        """Simple ConvNet with one 1D conv and pooling layer"""
        inputs = tf.keras.Input(shape=(8, 2), name="inputs")
        x = tf.keras.layers.Conv1D(2, 3, activation="relu", padding="same")(inputs)
        x = tf.keras.layers.MaxPooling1D(2)(x)
        # x = tf.keras.layers.Flatten()(x)
        # x = tf.keras.layers.Dense(10, activation="tanh")(x)
        return tf.keras.Model(inputs=inputs, outputs=x)

    @staticmethod
    def build_deep_conv():
        """Deeper ConvNet with multiple 1D conv and pooling layers"""
        inputs = tf.keras.Input(shape=(256, 3), name="inputs")
        x = tf.keras.layers.Conv1D(32, 3, activation="relu", padding="same")(inputs)
        x = tf.keras.layers.MaxPooling1D(2)(x)
        x = tf.keras.layers.Conv1D(64, 3, activation="relu", padding="same")(x)
        x = tf.keras.layers.MaxPooling1D(2)(x)
        x = tf.keras.layers.Conv1D(128, 3, activation="relu", padding="same")(x)
        # x = tf.keras.layers.AvgPool1D(pool_size=3, padding="same")(x)
        x = tf.keras.layers.Flatten()(x)
        outputs = tf.keras.layers.Dense(5, activation="sigmoid")(x)
        return tf.keras.Model(inputs=inputs, outputs=outputs)

    @staticmethod
    def build_conv_with_dropout():
        """ConvNet with 1D conv and dropout for regularization"""
        inputs = tf.keras.Input(shape=(100, 1), name="inputs")
        x = tf.keras.layers.Conv1D(32, 3, activation="relu")(inputs)
        x = tf.keras.layers.MaxPooling1D(2)(x)
        x = tf.keras.layers.Dropout(0.25)(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(64, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        outputs = tf.keras.layers.Dense(2, activation="softmax")(x)
        return tf.keras.Model(inputs=inputs, outputs=outputs)

    @staticmethod
    def build_conv_multi_output():
        """ConvNet with 1D conv and multiple outputs"""
        inputs = tf.keras.Input(shape=(128, 3), name="inputs")
        x = tf.keras.layers.Conv1D(16, 3, activation="relu", padding="same")(inputs)
        x = tf.keras.layers.MaxPooling1D(2)(x)
        x = tf.keras.layers.Flatten()(x)
        out1 = tf.keras.layers.Dense(4, activation="softmax", name="class_output")(x)
        out2 = tf.keras.layers.Dense(1, activation="sigmoid", name="score_output")(x)
        return tf.keras.Model(inputs=inputs, outputs={"class_output": out1, "score_output": out2})