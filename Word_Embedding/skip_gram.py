# -*- coding: utf-8 -*-
# @Time    : 19-1-15 上午6:42
# @Author  : zheng
# @Software: PyCharm
from collections import Counter
import tensorflow as tf
import numpy as np
import re
from utils.softmax import softmax

"""
思路：
实现根据输入单词与目标单词、输入单词与上下文单词得到代价与权重矩阵的梯度
"""

PARAMS = {
    'min_freq': 5,
    'skip_window': 5,
    'n_sampled': 100,
    'embed_dim': 200,
    'sample_words':['six', 'gold', 'japan', 'college'],
    'n_epochs': 10
}

def preprocess_text(text):
    text = text.replace('\n', " ")
    text = re.sub('\s+', ' ', text).strip().lower()

    words = text.split()
    word2freq = Counter(words)
    words = [word for word in words if word2freq[word] > PARAMS['min_freq']]
    print("Total words:",  len(words))

    _words = set(words)
    PARAMS['word2idx'] = {c: i for i, c in enumerate(_words)}
    PARAMS['idx2word'] = {i: c for i, c in enumerate(_words) }
    PARAMS['vocab_size'] = len(PARAMS['word2idx'])

    indexed = [PARAMS['word2idx'][w] for w in words]
    indexed = filter_high_freq(indexed)
    print("Word preprocessing completed...")

    return indexed

def filter_high_freq(int_words, t=1e-5, threshold=0.8):
    int_word_counts = Counter(int_words)
    total_count = len(int_words)

    word_freq = {w:c/total_count for w, c in int_word_counts.items()}
    prob_drop = {w: 1-np.sqrt(t / word_freq[w]) for w in int_word_counts}
    train_words = [w for w in int_word_counts if prob_drop[w] < threshold]
    return train_words






def make_data(int_words):
    x, y = [], []
    for i in range(len(int_words)):
        input_w = int_words[i]
        labels = get_y(int_words, i)
        x.extend([input_w] * len(labels))
        y.extend(labels)
    return x, y

def get_y(words, idx):
    skip_window = np.random.randint(1, PARAMS['skip_window']+1)
    left = idx - skip_window if (idx - skip_window) > 0 else 0
    right = idx + skip_window
    y = words[left:idx] + words[idx+1:right+1]
    return list(set(y))


def model_fn(features, labels, mode, params):
    w = tf.get_variable('softmax_w', [PARAMS['vocab_size'], PARAMS['embed_dim']])
    b = tf.get_variable('softmax_b', [PARAMS['vocab_size']])
    E = tf.get_variable('embeding', [PARAMS['vocab_size'], PARAMS['embed_dim']])

    embedded = tf.nn.embedding_lookup(E, features['x'])

    if mode == tf.estimator.ModeKeys.TRAIN:
        loss_op = tf.reduce_mean(tf.nn.sampled_softmax_loss(
            weights=w,
            biases=b,
            labels=labels,
            inputs=embedded,
            num_sampled=PARAMS['n_sampled'],
            num_classes=PARAMS['vocab_size']
        ))

        train_op = tf.train.AdamOptimizer().minimize(
            loss_op,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss_op, train_op=train_op)
    if mode == tf.estimator.ModeKeys.PREDICT:
        normalize_E = tf.nn.l2_normalize(E, -1)
        sample_E = tf.nn.embedding_lookup(normalize_E, features['x'])
        similarity = tf.matmul(sample_E, normalize_E, transpose_b=True)

        return tf.estimator.EstimatorSpec(mode, predictions=similarity)


with open("/home/andrea/PycharmProjects/NLP_Tasks/data/simple-examples/simple-examples/data/ptb.train.txt") as f:
    x_train, y_train = make_data(preprocess_text(f.read()))
estimator = tf.estimator.Estimator(model_fn)

