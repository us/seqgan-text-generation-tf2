# Discriminator and Generator Common Settings
EMB_DIM = 32  # embedding dimension
HIDDEN_DIM = 32  # hidden state dimension of lstm cell
SEQ_LENGTH = 20  # sequence length (words number)
MIN_SEQ_LENGTH = 10  # minimum sequence length
BATCH_SIZE = 64

# Discriminator Model Settings
dis_embedding_dim = 64
dis_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
dis_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]
dis_dropout_keep_prob = 0.75
dis_l2_reg_lambda = 0.2

# Epoch Number
PRE_EPOCH_NUM = 120
EPOCH_NUM = 40

generated_num = 10000
vocab_size = 20000

# Dataset
dataset_path = 'dataset/IMDB Dataset.csv'
positive_file = 'dataset/positives.txt'
negative_file = 'dataset/negatives.txt'
generated_file = 'dataset/generated_file.txt'

# Saved Models
pretrained_generator_file = "pretrained_models/pretrained_generator.h5"
pretrained_discriminator_file = "pretrained_models/pretrained_discriminator.h5"
generator_file = "pretrained_models/generator.h5"
discriminator_file = "pretrained_models/discriminator.h5"

tokenizer_file = 'pretrained_models/tokenizer.pickle'
