#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
#tf.disable_v2_behavior()

import os
import random
from dataloader import dataset_for_generator, dataset_for_discriminator
from generator import Generator
from discriminator import Discriminator
from rollout import ROLLOUT
from target_lstm import TARGET_LSTM
import pickle

#########################################################################################
#  生成器のパラメータ
######################################################################################
EMB_DIM = 32 # embedding dimension
HIDDEN_DIM = 32 # hidden state dimension of lstm cell
SEQ_LENGTH = 20 # sequence length
START_TOKEN = 0
PRE_EPOCH_NUM = 120 # supervise (maximum likelihood estimation) epochs
SEED = 88
BATCH_SIZE = 64

#########################################################################################
#  識別器のパラメータ
#########################################################################################
dis_embedding_dim = 64
dis_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
dis_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]
dis_dropout_keep_prob = 0.75
dis_l2_reg_lambda = 0.2
dis_batch_size = 64

#########################################################################################
#  Basic Training Parameters
# GANの学習を実行していく
#########################################################################################

TOTAL_BATCH = 200 # バッチサイズ

# 学習で使用するデータ
# 最初は存在しないので、lstmで作るらしい
positive_file = 'save/hakataeki_haiku2id.txt'
negative_file = 'save/generator_sample.txt'
# このファイルはどうした？
eval_file = 'save/eval_file.txt'

# 生成された数
generated_num = 10000

def main():
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    assert START_TOKEN == 0

    vocab_size = 5000

    physical_devices = tf.config.experimental.list_physical_devices("GPU")
    if len(physical_devices) > 0:
        for dev in physical_devices:
            tf.config.experimental.set_memory_growth(dev, True)

    # 生成器の初期設定
    generator = Generator(vocab_size, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH, START_TOKEN)
    # 最初のパラメータをpicklファイルから参照
    target_params = pickle.load(open('save/target_params_py3.pkl', 'rb'))
    target_lstm = TARGET_LSTM(BATCH_SIZE, SEQ_LENGTH, START_TOKEN, target_params) # The oracle model

    # 識別器の初期設定
    discriminator = Discriminator(sequence_length=SEQ_LENGTH, num_classes=2, vocab_size=vocab_size, embedding_size=dis_embedding_dim,
                                  filter_sizes=dis_filter_sizes, num_filters=dis_num_filters, dropout_keep_prob=dis_dropout_keep_prob,
                                  l2_reg_lambda=dis_l2_reg_lambda)

    # First, use the oracle model to provide the positive examples, which are sampled from the oracle data distribution
    # GANの学習で使用する正解データを作成する
    if not os.path.exists(positive_file):
        target_lstm.generate_samples(generated_num // BATCH_SIZE, positive_file)
    gen_dataset = dataset_for_generator(positive_file, BATCH_SIZE)
    log = open('save/experiment-log.txt', 'w')


    #  事前学習での文章生成をlstmで行い、生成器の重みを保存する
    if not os.path.exists("generator_pretrained.h5"):
        print('Start pre-training...')
        log.write('pre-training...\n')
        generator.pretrain(gen_dataset, target_lstm, PRE_EPOCH_NUM, generated_num // BATCH_SIZE, eval_file)
        generator.save("generator_pretrained.h5")
    else:
        generator.load("generator_pretrained.h5")

    # 識別器の事前学習での重み
    if not os.path.exists("discriminator_pretrained.h5"):
        print('Start pre-training discriminator...')
        # Train 3 epoch on the generated data and do this for 50 times
        # 3エポックの識別器の訓練を５０回繰り返す
        for _ in range(50):
            print("Dataset", _)

            # まず生成器が偽物を作成
            generator.generate_samples(generated_num // BATCH_SIZE, negative_file)

            # 偽物と本物を混ぜたデータセットを作成
            dis_dataset = dataset_for_discriminator(positive_file, negative_file, BATCH_SIZE)

            # 識別器を学習させる
            discriminator.train(dis_dataset, 3, (generated_num // BATCH_SIZE) * 2)
        discriminator.save("discriminator_pretrained.h5")
    else:
        discriminator.load("discriminator_pretrained.h5")

    rollout = ROLLOUT(generator, 0.8)

    print('#########################################################################')
    print('Start Adversarial Training...')
    log.write('adversarial training...\n')

    # 学習の実行
    # 今回は200回の訓練を行う
    for total_batch in range(TOTAL_BATCH):
        print("Generator", total_batch)
        # Train the generator for one step
        for it in range(1):
            samples = generator.generate_one_batch()
            rewards = rollout.get_reward(samples, 16, discriminator)
            generator.train_step(samples, rewards)

        # Test
        if total_batch % 5 == 0 or total_batch == TOTAL_BATCH - 1:
            generator.generate_samples(generated_num // BATCH_SIZE, eval_file)
            likelihood_dataset = dataset_for_generator(eval_file, BATCH_SIZE)
            test_loss = target_lstm.target_loss(likelihood_dataset)
            buffer = 'epoch:\t' + str(total_batch) + '\tnll:\t' + str(test_loss) + '\n'
            print('total_batch: ', total_batch, 'test_loss: ', test_loss)
            log.write(buffer)

        # Update roll-out parameters
        rollout.update_params()

        # Train the discriminator
        print("Discriminator", total_batch)
        for _ in range(5):
            generator.generate_samples(generated_num // BATCH_SIZE, negative_file)
            dis_dataset = dataset_for_discriminator(positive_file, negative_file, BATCH_SIZE)
            discriminator.train(dis_dataset, 3, (generated_num // BATCH_SIZE) * 2)
    generator.save("generator.h5")
    discriminator.save("discriminator.h5")

    log.close()


if __name__ == '__main__':
    main()
