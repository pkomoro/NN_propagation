import logging

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

import lightprop.propagation.methods as prop
from lightprop.lightfield import LightField


class NNTrainer:
    def __init__(self):
        self.model = None
        self.history = None
        self.log = logging.getLogger(type(self).__name__)

    def amplitudeMSE(self, y_true, y_pred):
        squared_difference = tf.square(y_true[0, 0] - y_pred[0, 0])

        return tf.reduce_mean(squared_difference, axis=-1)

    def intensityMSE(self, y_true, y_pred):
        squared_difference = tf.square(tf.square(y_true[0, 0]) - tf.square(y_pred[0, 0]))

        return tf.reduce_mean(squared_difference, axis=-1)

    def optimize(self, input_field: LightField, target_field: LightField, distance, iterations: int = 100):
        propagator = prop.NNPropagation()
        self.model = propagator.build_model(input_field.matrix_size)

        self.model = propagator.set_kernels(self.model, input_field, distance)

        self.log.info("Compiling model...")
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-1,
            decay_steps=1000,
            decay_rate=0.9)
        self.model.compile(
            optimizer=keras.optimizers.Adam(),
            loss=self.intensityMSE,
        )

        checkpoint_filepath = "./tmp/checkpoint"
        self.log.info(f"Setting up checkpoint at {checkpoint_filepath}...")
        model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath, save_weights_only=True, monitor="loss", mode="min", save_best_only=True
        )

        self.log.info("Fitting model...")
        self.history = self.model.fit(
            propagator.prepare_input_field(input_field),
            propagator.prepare_input_field(target_field),
            batch_size=1,
            epochs=iterations,
            callbacks=[model_checkpoint_callback],
        )

        self.log.info("Loading best configuration...")
        self.model.load_weights(checkpoint_filepath)
        return self.model

    def plot_loss(self, path):
        data = self.history.history["loss"]
        plt.plot(data)
        plt.title("model loss")
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.yscale("log")
        plt.savefig(path + '.jpg', dpi=1000, bbox_inches='tight')
        with open(path + '.txt', 'w') as file:
            for line in data:
                file.write(f"{str(line)}\n")
        # plt.show()


class NNMultiTrainer(NNTrainer):
    def optimize(
        self, input_field: LightField, target_field: LightField, kernel: LightField, distance, iterations: int = 100
    ):
        propagator = prop.MultiparameterNNPropagation()
        self.model = propagator.build_model(input_field.matrix_size)

        self.log.info("Compiling model...")
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-1),
            loss=keras.losses.MeanSquaredError(),
        )

        checkpoint_filepath = "./tmp/checkpoint"
        self.log.info(f"Setting up checkpoint at {checkpoint_filepath}...")
        model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath, save_weights_only=True, monitor="loss", mode="min", save_best_only=True
        )

        self.log.info("Fitting model...")
        self.history = self.model.fit(
            [propagator.prepare_input_field(input_field), propagator.prepare_input_field(kernel)],
            propagator.prepare_input_field(target_field),
            batch_size=1,
            epochs=iterations,
            callbacks=[model_checkpoint_callback],
        )

        self.log.info("Loading best configuration...")
        self.model.load_weights(checkpoint_filepath)
        return self.model


class NN_FFTTrainer(NNTrainer):
    def optimize(
        self, input_field: LightField, target_field: LightField, kernel: LightField, wavelength_scaling, phase_map, iterations: int = 100):
        propagator = prop.MultiparameterNNPropagation_FFTConv()
        self.model = propagator.build_model(input_field[0].matrix_size, phase_map)

        self.log.info("Compiling model...")
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=5e-3),
            loss=keras.losses.MeanSquaredError(),
        )

        checkpoint_filepath = "./tmp/checkpoint"
        self.log.info(f"Setting up checkpoint at {checkpoint_filepath}...")
        model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath, save_weights_only=True, monitor="loss", mode="min", save_best_only=True
        )

        self.log.info("Fitting model...")
        self.history = self.model.fit(
            [
                np.array(list(map(propagator.prepare_input_field, input_field))).reshape(
                    (len(input_field), 2, input_field[0].matrix_size, input_field[0].matrix_size)
                ),
                np.array(list(map(propagator.prepare_input_field, kernel))).reshape(
                    (len(input_field), 2, input_field[0].matrix_size, input_field[0].matrix_size)
                ),
                np.array(wavelength_scaling).reshape(len(input_field),),
            ],
            np.array(list(map(propagator.prepare_input_field, target_field))).reshape(
                (len(input_field), 2, input_field[0].matrix_size, input_field[0].matrix_size)
            ),
            batch_size=1,
            epochs=iterations,
            callbacks=[model_checkpoint_callback],
        )

        self.log.info("Loading best configuration...")
        self.model.load_weights(checkpoint_filepath)
        return self.model
