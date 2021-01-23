import pandas as pd
import string
from nltk.tokenize import word_tokenize, sent_tokenize
import re
import tensorflow as tf
import pickle
import numpy as np

TAG_RE = re.compile(r'<[^>]+>')


def remove_tags(text):
    return TAG_RE.sub('', text)


def clean_text(text):
    def tokenize_text(text):
        return [w for s in sent_tokenize(text) for w in word_tokenize(s)]

    def remove_special_characters(text, characters=string.punctuation.replace('-', '')):
        tokens = tokenize_text(text)
        pattern = re.compile('[{}]'.format(re.escape(characters)))
        return ' '.join(filter(None, [pattern.sub('', t) for t in tokens]))

    text = text.strip(' ')  # strip whitespaces
    text = remove_tags(text)
    text = text.lower()  # lowercase
    text = remove_special_characters(text)  # remove punctuation and symbols
    return text


# print(clean_text('hey you mother fuckerb,<br> , ,, !hopo ! seixasga sd  a'))
df = pd.read_csv('dataset/IMDB Dataset.csv')

data_num = 10000

df = df.review[:data_num]


def split_and_clean(data):
    arr = []
    for i in data:
        sentences = i.split('.')
        for s in sentences:
            if len(s.split()) < 21:
                if s:
                    arr.append(s)
    for i, j in enumerate(arr):
        arr[i] = clean_text(arr[i])
    return arr


def tokenize(arr):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    tokenizer.fit_on_texts(arr)
    tensor = tokenizer.texts_to_sequences(arr)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,
                                                           padding='post', maxlen=20)

    # saving
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return tensor

    # # loading
    # with open('tokenizer.pickle', 'rb') as handle:
    #     tokenizer = pickle.load(handle)

    # print(tokenizer.sequences_to_texts([[2, 5, 3, 0, 1]]))


d = split_and_clean(df)
d = tokenize(d)
print(d.shape)

np.savetxt('dataset/positives.txt', d[:10000], delimiter=' ', fmt='%i')  # X is an array
np.savetxt('dataset/negatives.txt', d[10000:20000], delimiter=' ', fmt='%i')  # X is an array
np.savetxt('dataset/evals.txt', d[20000:21000], delimiter=' ', fmt='%i')  # X is an array