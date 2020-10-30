import tensorflow as tf
import numpy as np
from rnnlm import RNNLM

class ROLLOUT(RNNLM):
    def __init__(self, lstm, update_rate):
        super(ROLLOUT, self).__init__(lstm.num_emb, lstm.batch_size, lstm.emb_dim, lstm.hidden_dim, lstm.sequence_length, lstm.start_token)
        self.lstm = lstm
        self.update_rate = update_rate
        self.g_model.set_weights(lstm.g_model.get_weights())

    @tf.function
    def generate_one_batch(self, x_orig, given_num):
        # Initial states
        h0 = c0 = tf.zeros([self.batch_size, self.hidden_dim])
        h0 = [h0, c0]
        processed_x = tf.transpose(tf.nn.embedding_lookup(self.g_embeddings, x_orig), perm=[1, 0, 2])  # seq_length x batch_size x emb_dim
        gen_x = tf.TensorArray(dtype=tf.int32, size=self.sequence_length,
                                             dynamic_size=False, infer_shape=True)

        ta_emb_x = tf.TensorArray(dtype=tf.float32, size=self.sequence_length)
        ta_emb_x = ta_emb_x.unstack(processed_x)
        ta_x = tf.TensorArray(dtype=tf.int32, size=self.sequence_length)
        ta_x = ta_x.unstack(tf.transpose(x_orig, perm=[1, 0]))

        # When current index i < given_num, use the provided tokens as the input at each time step
        def _g_recurrence_1(i, x_t, h_tm1, given_num, gen_x):
            # h_t: hidden_memory_tuple
            _, h_t = self.g_model.layers[1].cell(x_t, h_tm1, training=False) # layers[1]: LSTM
            x_tp1 = ta_emb_x.read(i)
            next_token = ta_x.read(i)
            gen_x = gen_x.write(i, next_token)  # indices, batch_size
            return i + 1, x_tp1, h_t, given_num, gen_x

        # When current index i >= given_num, start roll-out, use the output as time step t as the input at time step t+1
        def _g_recurrence_2(i, x_t, h_tm1, given_num, gen_x):
            # o_t: batch x vocab, probability
            # h_t: hidden_memory_tuple
            o_t, h_t = self.g_model.layers[1].cell(x_t, h_tm1, training=False) # layers[1]: LSTM
            o_t = self.g_model.layers[2](o_t) # layers[2]: Dense
            log_prob = tf.math.log(o_t)
            next_token = tf.cast(tf.reshape(tf.random.categorical(log_prob, 1), [self.batch_size]), tf.int32)
            x_tp1 = tf.nn.embedding_lookup(self.g_embeddings, next_token)  # batch x emb_dim
            gen_x = gen_x.write(i, next_token)  # indices, batch_size
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

        # gen_x: seq_length x batch_size
        outputs = tf.transpose(gen_x.stack(), perm=[1, 0])  # batch_size x seq_length
        return outputs

    def get_reward(self, input_x, rollout_num, discriminator):
        rewards = []
        for i in range(rollout_num):
            # given_num between 1 to sequence_length - 1 for a part completed sentence
            for given_num in tf.range(1, self.sequence_length):
                samples = self.generate_one_batch(input_x, given_num)
                ypred_for_auc = discriminator.d_model(samples).numpy()
                ypred = ypred_for_auc[:, 1] # prob for outputting 1 (True)
                if i == 0:
                    rewards.append(ypred)
                else:
                    rewards[given_num - 1] += ypred

            # the last token reward
            ypred_for_auc = discriminator.d_model(input_x).numpy()
            ypred = ypred_for_auc[:, 1]
            if i == 0:
                rewards.append(ypred)
            else:
                # completed sentence reward
                rewards[self.sequence_length - 1] += ypred

        rewards = np.transpose(np.array(rewards)) / (1.0 * rollout_num)  # batch_size x seq_length
        return rewards

    def update_params(self):
        # Weights and Bias for input and hidden tensor
        # self: Rollout, self.lstm: Original generator

        # The embedding layer: directly (fully) transferred
        # The other layers: transferred with the ratio of (1 - self.update_rate)
        new_weights = [self.update_rate * w1 + (1 - self.update_rate) * w2 if i > 0 else w2
                       for i, (w1, w2) in enumerate(zip(self.g_model.get_weights(), self.lstm.g_model.get_weights()))]
        self.g_model.set_weights(new_weights)
