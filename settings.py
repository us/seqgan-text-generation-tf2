EMB_DIM = 32  # embedding dimension
HIDDEN_DIM = 32  # hidden state dimension of lstm cell
SEQ_LENGTH = 20  # sequence length (words number)
BATCH_SIZE = 64

# Discriminator Model Settings
dis_embedding_dim = 64
dis_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
dis_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]
dis_dropout_keep_prob = 0.75
dis_l2_reg_lambda = 0.2

# Epoch Number
EPOCH_NUM = 15
PRE_EPOCH_NUM = 40

positive_file = 'dataset/positives.txt'
negative_file = 'dataset/negatives.txt'
eval_file = 'dataset/evals.txt'
output_file = 'dataset/output_file.txt'
tokenizer_file = 'pretrained_models/tokenizer.pickle'
dataset_path = 'dataset/IMDB Dataset.csv'

generated_num = 10000
vocab_size = 20000
