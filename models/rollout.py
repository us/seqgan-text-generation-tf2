import numpy as np
import tensorflow as tf

from models.rnn import RNN


class ROLLOUT(RNN):
    def __init__(self, lstm, update_rate):
        super(ROLLOUT, self).__init__(lstm.num_emb, lstm.batch_size, lstm.emb_dim, lstm.hidden_dim,
                                      lstm.sequence_length, lstm.start_token)
        self.lstm = lstm
        self.update_rate = update_rate
        self.generator_model.set_weights(lstm.generator_model.get_weights())

    @tf.function
    def generate_one_batch(self, x_orig, given_num):
        h0 = c0 = tf.zeros([self.batch_size, self.hidden_dim])
        h0 = [h0, c0]
        processed_x = tf.transpose(tf.nn.embedding_lookup(self.g_embeddings, x_orig),
                                   perm=[1, 0, 2])
        gen_x = tf.TensorArray(dtype=tf.int32, size=self.sequence_length,
                               dynamic_size=False, infer_shape=True)

        ta_emb_x = tf.TensorArray(dtype=tf.float32, size=self.sequence_length)
        ta_emb_x = ta_emb_x.unstack(processed_x)
        ta_x = tf.TensorArray(dtype=tf.int32, size=self.sequence_length)
        ta_x = ta_x.unstack(tf.transpose(x_orig, perm=[1, 0]))

        def _g_recurrence_1(i, x_t, h_tm1, given_num, gen_x):
            _, h_t = self.generator_model.layers[1].cell(x_t, h_tm1, training=False)
            x_tp1 = ta_emb_x.read(i)
            next_token = ta_x.read(i)
            gen_x = gen_x.write(i, next_token)
            return i + 1, x_tp1, h_t, given_num, gen_x

        def _g_recurrence_2(i, x_t, h_tm1, given_num, gen_x):
            o_t, h_t = self.generator_model.layers[1].cell(x_t, h_tm1, training=False)
            o_t = self.generator_model.layers[2](o_t)
            log_prob = tf.math.log(o_t)
            next_token = tf.cast(tf.reshape(tf.random.categorical(log_prob, 1), [self.batch_size]), tf.int32)
            x_tp1 = tf.nn.embedding_lookup(self.g_embeddings, next_token)
            gen_x = gen_x.write(i, next_token)
            return i + 1, x_tp1, h_t, given_num, gen_x

        i, x_t, h_tm1, given_num, gen_x = tf.while_loop(
            cond=lambda i, _1, _2, given_num, _4: i < given_num,
            body=_g_recurrence_1,
            loop_vars=(tf.constant(0, dtype=tf.int32),
                       tf.nn.embedding_lookup(self.g_embeddings, self.start_token_vec), h0, given_num, gen_x))

        _, _, _, _, gen_x = tf.while_loop(
            cond=lambda i, _1, _2, _3, _4: i < self.sequence_length,
            body=_g_recurrence_2,
            loop_vars=(i, x_t, h_tm1, given_num, gen_x))

        outputs = tf.transpose(gen_x.stack(), perm=[1, 0])
        return outputs

    def get_reward(self, input_x, rollout_num, discriminator):
        rewards = []
        for i in range(rollout_num):
            for given_num in tf.range(1, self.sequence_length):
                samples = self.generate_one_batch(input_x, given_num)
                ypred_for_auc = discriminator.d_model(samples).numpy()
                ypred = ypred_for_auc[:, 1]
                if i == 0:
                    rewards.append(ypred)
                else:
                    rewards[given_num - 1] += ypred

            ypred_for_auc = discriminator.d_model(input_x).numpy()
            ypred = ypred_for_auc[:, 1]
            if i == 0:
                rewards.append(ypred)
            else:
                rewards[self.sequence_length - 1] += ypred

        rewards = np.transpose(np.array(rewards)) / (1.0 * rollout_num)  # batch_size x seq_length
        return rewards

    def update_params(self):
        new_weights = [self.update_rate * w1 + (1 - self.update_rate) * w2 if i > 0 else w2
                       for i, (w1, w2) in
                       enumerate(zip(self.generator_model.get_weights(), self.lstm.generator_model.get_weights()))]
        self.generator_model.set_weights(new_weights)
