from models.rnn import RNN

import numpy as np
import tensorflow as tf


class Generator(RNN):
    def __init__(self, num_emb, batch_size, emb_dim, hidden_dim, sequence_length, learning_rate=0.01):
        super(Generator, self).__init__(num_emb, batch_size, emb_dim, hidden_dim, sequence_length, learning_rate)

        self.generator_model = tf.keras.models.Sequential(self.generator_model.layers)
        self.generator_optimizer = self._create_optimizer(
            learning_rate,
            clipnorm=self.grad_clip
        )
        self.generator_model.compile(
            optimizer=self.generator_optimizer,
            loss="sparse_categorical_crossentropy",
            sample_weight_mode="temporal")

    def pretrain(self, dataset, num_epochs, num_steps):
        ds = dataset.map(lambda x: (tf.pad(x[:, 0:-1], ([0, 0], [1, 0]), "CONSTANT", 0), x)).repeat(
            num_epochs)
        pretrain_loss = self.generator_model.fit(ds, verbose=1, epochs=num_epochs, steps_per_epoch=num_steps)
        print("Pretrain generator loss: ", pretrain_loss)
        return pretrain_loss

    def train_step(self, x, rewards):
        train_loss = self.generator_model.train_on_batch(
            np.pad(x[:, 0:-1], ([0, 0], [1, 0]), "constant", constant_values=0),
            x,
            sample_weight=rewards * self.batch_size * self.sequence_length
        )
        print("Generator Loss: ", train_loss)
        return train_loss

    def _create_optimizer(self, *args, **kwargs):
        return tf.keras.optimizers.Adam(*args, **kwargs)

    def save(self, filename):
        self.generator_model.save_weights(filename, save_format="h5")

    def load(self, filename):
        self.generator_model.load_weights(filename)
