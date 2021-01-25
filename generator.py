import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Dense, LSTM
from tensorflow.keras.layers import Input, Embedding, Dense, LSTM
import numpy as np
from rnnlm import RNNLM
from dataloader import dataset_for_generator

class Generator(RNNLM):
    def __init__(self, num_emb, batch_size, emb_dim, hidden_dim, sequence_length, start_token, learning_rate=0.01):
        super(Generator, self).__init__(num_emb, batch_size, emb_dim, hidden_dim, sequence_length, start_token, learning_rate)
        
        # prepare model for GAN training
        # GANのモデルを作成
        self.g_model_temporal = tf.keras.models.Sequential(self.g_model.layers)
        self.g_optimizer_temporal = self._create_optimizer(
            learning_rate, clipnorm=self.grad_clip)
        self.g_model_temporal.compile(
            optimizer=self.g_optimizer_temporal,
            loss="sparse_categorical_crossentropy",
            sample_weight_mode="temporal")

# 事前学習用のモデル
    def pretrain(self, dataset, num_epochs, num_steps):
        # dataset: each element has [self.batch_size, self.sequence_length]
        # outputs are 1 timestep ahead
        
        # 途中経過を表示
        def pretrain_callback(epoch, logs):
            if epoch % 5 == 0:
                # self.generate_samples(num_steps, eval_file)
                # likelihood_dataset = dataset_for_generator(eval_file, self.batch_size)
                # test_loss = target_lstm.target_loss(likelihood_dataset)
                print('pre-train epoch ', epoch, 'test_loss ', "-")
                buffer = 'epoch:\t'+ str(epoch) + '\tnll:\t' + "-" + '\n'
                logs.write(buffer)

        ds = dataset.map(lambda x: (tf.pad(x[:, 0:-1], ([0, 0], [1, 0]), "CONSTANT", self.start_token), x)).repeat(num_epochs)
        # 事前学習の実行
        pretrain_loss = self.g_model.fit(ds, verbose=1, epochs=num_epochs, steps_per_epoch=num_steps,
                                         callbacks=[tf.keras.callbacks.LambdaCallback(on_epoch_end=pretrain_callback)])
        return pretrain_loss

 # GANの生成器のモデル
    def train_step(self, x, rewards):
        # x: [self.batch_size, self.sequence_length]
        # rewards: [self.batch_size, self.sequence_length] (sample_weight)
        # outputs are 1 timestep ahead
        train_loss = self.g_model_temporal.train_on_batch(
            np.pad(x[:, 0:-1], ([0, 0], [1, 0]), "constant", constant_values=self.start_token), x,
            # sparse_categorical_crossentropy returns mean loss
            # here we multiply (batch_size * sequence_length) to use weighted "sum"
            sample_weight=rewards * self.batch_size * self.sequence_length)
        return train_loss

    def _create_optimizer(self, *args, **kwargs):
        return tf.keras.optimizers.Adam(*args, **kwargs)

    def save(self, filename):
        self.g_model.save_weights(filename, save_format="h5")

    def load(self, filename):
        self.g_model.load_weights(filename)
