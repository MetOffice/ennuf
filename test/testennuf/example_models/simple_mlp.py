#  (C) Crown Copyright, Met Office, 2023.
import tensorflow as tf
import tensorflow as tf


class SimpleMLP:
    @staticmethod
    def build_functional_easy():
        """Builds some machine learning model with the keras functional API"""
        nnodes = 8
        inputs = tf.keras.Input(shape=6, name="inputs")
        y = tf.keras.layers.Dense(nnodes, activation="relu", name="dense")(inputs)
        outputs = tf.keras.layers.Dense(4, name="outputs")(y)
        return tf.keras.models.Model(inputs=inputs, outputs=outputs)

    @staticmethod
    def build_functional():
        """Builds some machine learning model with the keras functional API"""
        nnodes = 8
        activation = "relu"
        reg = None
        scalars = tf.keras.Input(shape=6, name="scalars")
        y = tf.keras.layers.Dense(
            nnodes, activation=activation, kernel_regularizer=reg, name="dense1"
        )(scalars)
        y = tf.keras.layers.Dense(
            nnodes, activation="sigmoid", kernel_regularizer=reg, name="dense2"
        )(y)
        y = tf.keras.layers.Dense(
            nnodes, activation="relu", kernel_regularizer=reg, name="dense3"
        )(y)
        y = tf.keras.layers.Dense(
            nnodes, activation="tanh", kernel_regularizer=reg, name="dense4"
        )(y)
        outputs_1 = tf.keras.layers.Dense(1, name="outputs_1")(y)
        outputs_2 = tf.keras.layers.Dense(2, name="outputs_2")(y)
        return tf.keras.models.Model(
            inputs=scalars, outputs={"outputs_1": outputs_1, "outputs_2": outputs_2}
        )

    @staticmethod
    def build_functional_1_1():
        """Builds some machine learning model with the keras functional API"""
        nnodes = 8
        activation = "relu"
        reg = None
        scalars = tf.keras.Input(shape=6, name="scalars")
        y = tf.keras.layers.Dense(
            nnodes, activation=activation, kernel_regularizer=reg, name="dense1"
        )(scalars)
        y = tf.keras.layers.Dense(
            nnodes, activation="sigmoid", kernel_regularizer=reg, name="dense2"
        )(y)
        y = tf.keras.layers.Dense(
            nnodes, activation="relu", kernel_regularizer=reg, name="dense3"
        )(y)
        y = tf.keras.layers.Dense(
            nnodes, activation="tanh", kernel_regularizer=reg, name="dense4"
        )(y)
        outputs_1 = tf.keras.layers.Dense(1, name="outputs_1")(y)
        outputs_2 = tf.keras.layers.Dense(2, name="outputs_2")(y)
        return tf.keras.models.Model(
            inputs=scalars, outputs={"foo": outputs_1, "bar": outputs_2}
        )

    @staticmethod
    def build_functional_2():
        """Builds some machine learning model with the keras functional API"""
        nnodes = 32
        nnodes2 = 16
        activation = "relu"
        reg = None
        scalars = tf.keras.Input(shape=6, name="scalars")
        profile = tf.keras.Input(shape=70, name="profile")
        y = tf.keras.layers.Dense(
            nnodes, activation=activation, kernel_regularizer=reg, name="densey1"
        )(scalars)
        y = tf.keras.layers.Dense(
            nnodes2, activation="sigmoid", kernel_regularizer=reg, name="densey2"
        )(y)
        y = tf.keras.layers.Dense(
            nnodes, activation="relu", kernel_regularizer=reg, name="densey3"
        )(y)
        y = tf.keras.layers.Dense(
            nnodes2, activation="tanh", kernel_regularizer=reg, name="densey4"
        )(y)
        z = tf.keras.layers.Dense(
            nnodes2, activation=activation, kernel_regularizer=reg, name="densez1"
        )(profile)
        z = tf.keras.layers.Dense(
            nnodes, activation=None, kernel_regularizer=reg, name="densez2"
        )(z)
        z = tf.keras.layers.Dense(
            nnodes2, activation="relu", kernel_regularizer=reg, name="densez3"
        )(z)
        z = tf.keras.layers.Dense(
            nnodes, activation="tanh", kernel_regularizer=reg, name="densez4"
        )(z)
        outputs_1 = tf.keras.layers.Dense(9, name="outputs_1")(y)
        outputs_2 = tf.keras.layers.Dense(2, name="outputs_2")(y)
        outputs_3 = tf.keras.layers.Dense(3, name="outputs_3")(z)
        return tf.keras.models.Model(
            inputs={"scalars": scalars, "profile": profile},
            outputs={
                "outputs_1": outputs_1,
                "outputs_2": outputs_2,
                "outputs_3": outputs_3,
            },
        )

    @staticmethod
    def build_sequential():
        nnodes = 8
        model = tf.keras.Sequential()
        model.add(tf.keras.Input(6))
        for activation in ["relu", "sigmoid", "tanh"]:
            model.add(tf.keras.layers.Dense(nnodes, activation=activation))
        return model.compile()

    @staticmethod
    def build_easy_covariance_predictor_separate_outputs(alpha=0.05):
        """Functional API model, entirely dense"""
        # Input reshaping
        inputs = tf.keras.Input(shape=24, name='inputs')
        activation = tf.keras.layers.LeakyReLU(alpha=alpha)
        nnodes = 4
        reg = None
        y = tf.keras.layers.Dense(nnodes, activation=activation, kernel_regularizer=reg, name='dense1')(inputs)
        y = tf.keras.layers.Dense(nnodes, activation=activation, kernel_regularizer=reg, name='dense2')(y)

        z = tf.keras.layers.Dense(nnodes, activation=activation, kernel_regularizer=reg, name='covdense1')(inputs)
        z = tf.keras.layers.Dense(nnodes, activation=activation, kernel_regularizer=reg, name='covdense2')(z)

        stddev = tf.keras.layers.Dense(7, activation='relu', name='stddevs')(y)
        covariances = tf.keras.layers.Dense(6, name='covariances')(z)
        return tf.keras.models.Model(inputs=inputs, outputs={'stddevs': stddev, 'covariances': covariances})

    @staticmethod
    def build_covariance_predictor_separate_outputs(alpha=0.05):
        """Functional API model, entirely dense"""
        # Input reshaping
        inputs = tf.keras.Input(shape=24, name='inputs')
        activation = tf.keras.layers.LeakyReLU(alpha=alpha)
        nnodes = 256
        reg = None
        y = tf.keras.layers.Dense(nnodes, activation=activation, kernel_regularizer=reg, name='dense1')(inputs)
        y = tf.keras.layers.Dense(nnodes, activation=activation, kernel_regularizer=reg, name='dense2')(y)
        y = tf.keras.layers.Dense(nnodes, activation=activation, kernel_regularizer=reg, name='dense3')(y)
        y = tf.keras.layers.Dense(nnodes, activation=activation, kernel_regularizer=reg, name='dense4')(y)

        z = tf.keras.layers.Dense(nnodes, activation=activation, kernel_regularizer=reg, name='covdense1')(inputs)
        z = tf.keras.layers.Dense(nnodes, activation=activation, kernel_regularizer=reg, name='covdense2')(z)
        z = tf.keras.layers.Dense(nnodes, activation=activation, kernel_regularizer=reg, name='covdense3')(z)
        z = tf.keras.layers.Dense(nnodes, activation=activation, kernel_regularizer=reg, name='covdense4')(z)

        stddev = tf.keras.layers.Dense(70, activation='relu', name='stddevs')(y)
        covariances = tf.keras.layers.Dense(69 + 50 + 9 + 8 + 7 + 6, name='covariances')(z)
        return tf.keras.models.Model(inputs=inputs, outputs={'stddevs': stddev, 'covariances': covariances})

    @staticmethod
    def build_covariance_predictor(alpha=0.1):
        """Functional API model, entirely dense"""
        # Input reshaping
        inputs = tf.keras.Input(shape=24, name='inputs')
        activation = tf.keras.layers.LeakyReLU(alpha=alpha)
        nnodes = 256
        reg = None
        y = tf.keras.layers.Dense(nnodes, activation=activation, kernel_regularizer=reg, name='dense1')(inputs)
        y = tf.keras.layers.Dense(nnodes, activation=activation, kernel_regularizer=reg, name='dense2')(y)
        y = tf.keras.layers.Dense(nnodes, activation=activation, kernel_regularizer=reg, name='dense3')(y)
        y = tf.keras.layers.Dense(nnodes, activation=activation, kernel_regularizer=reg, name='dense4')(y)

        z = tf.keras.layers.Dense(nnodes, activation=activation, kernel_regularizer=reg, name='covdense1')(inputs)
        z = tf.keras.layers.Dense(nnodes, activation=activation, kernel_regularizer=reg, name='covdense2')(z)
        z = tf.keras.layers.Dense(nnodes, activation=activation, kernel_regularizer=reg, name='covdense3')(z)
        z = tf.keras.layers.Dense(nnodes, activation=activation, kernel_regularizer=reg, name='covdense4')(z)

        stddev = tf.keras.layers.Dense(70, activation='relu', name='stddevs')(y)
        covariances = tf.keras.layers.Dense(69 + 50 + 9 + 8 + 7 + 6, name='covariances')(z)
        outputs = tf.keras.layers.Concatenate(name='outputs')([stddev, covariances])
        return tf.keras.models.Model(inputs=inputs, outputs=outputs)
