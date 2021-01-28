import tensorflow as tf
from models.generator import Generator
from settings import *


physical_devices = tf.config.experimental.list_physical_devices("GPU")
if len(physical_devices) > 0:
    for dev in physical_devices:
        tf.config.experimental.set_memory_growth(dev, True)


generator = Generator(vocab_size, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH)

generator.load("pretrained_models/generator.h5")
print(generator.generate_one_batch())
