import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Embedding, Conv1D, MaxPool1D, Concatenate, Flatten, Dropout
from tensorflow.keras.models import Model


class Highway(tf.keras.models.Model):
    def __init__(self, **kwargs):
        super(Highway, self).__init__(**kwargs)

    def build(self, input_shape):
        output_dim = input_shape[-1]
        self.dense_g = Dense(output_dim, activation="relu")
        self.dense_t = Dense(output_dim, activation="sigmoid")

    def call(self, input_tensor, training=False):
        g = self.dense_g(input_tensor, training=training)
        t = self.dense_t(input_tensor, training=training)
        o = t * g + (1. - t) * input_tensor
        return o


class Discriminator:
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """

    def __init__(
            self, sequence_length, num_classes, vocab_size,
            embedding_size, filter_sizes, num_filters, dropout_keep_prob, l2_reg_lambda=0.0):
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

        layer_input = Input((sequence_length,), dtype=tf.int32)
        layer_emb = Embedding(vocab_size, embedding_size,
                              embeddings_initializer=tf.random_uniform_initializer(-1.0, 1.0))(layer_input)

        pooled_outputs = []
        for filter_size, num_filter in zip(filter_sizes, num_filters):
            x = Conv1D(num_filter, filter_size)(layer_emb)
            x = MaxPool1D(sequence_length - filter_size + 1)(x)
            pooled_outputs.append(x)

        x = Concatenate()(pooled_outputs)
        x = Flatten()(x)
        x = Highway()(x)
        x = Dropout(1.0 - dropout_keep_prob)(x)
        layer_output = Dense(num_classes,
                             kernel_regularizer=tf.keras.regularizers.l2(l2_reg_lambda),
                             bias_regularizer=tf.keras.regularizers.l2(l2_reg_lambda),
                             activation="softmax")(x)

        self.d_model = Model(layer_input, layer_output)
        d_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.d_model.compile(optimizer=d_optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    # 学習の実行
    def train(self, dataset, num_epochs, num_steps, **kwargs):
        return self.d_model.fit(dataset.repeat(num_epochs), verbose=1, epochs=num_epochs, steps_per_epoch=num_steps,
                                **kwargs)

    def save(self, filename):
        self.d_model.save_weights(filename, save_format="h5")

    def load(self, filename):
        self.d_model.load_weights(filename)
