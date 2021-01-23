import numpy as np
from rnnlm import RNNLM


class TARGET_LSTM(RNNLM):
    # function for reading save/target_params.pkl
    # save/target_params.pklを読み込むための機能
    def __init__(self, batch_size, sequence_length, start_token, params):
        # Model sizes are determined by the parameter file
        # モデルの大きさはパラメータファイルで決められる
        num_emb = params[0].shape[0]
        emb_dim = params[0].shape[1]
        hidden_dim = params[1].shape[1]

        super(TARGET_LSTM, self).__init__(num_emb, batch_size, emb_dim, hidden_dim, sequence_length, start_token)
        weights = [
            # Embedding
            params[0],
            # LSTM
            np.c_[params[1], params[4], params[10], params[7]],  # kernel (i, f, c, o)
            np.c_[params[2], params[5], params[11], params[8]],  # recurrent_kernel
            np.r_[params[3], params[6], params[12], params[9]],  # bias
            # Dense
            params[13],
            params[14]
        ]
        self.g_model.set_weights(weights)
