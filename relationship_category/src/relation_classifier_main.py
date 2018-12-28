#!/usr/bin/python
#coding:utf-8
import tensorflow as tf
from relationship_category.src.configure import FLAGS
from relationship_category.src.att_lstm import AttLSTM
import numpy as np
from relationship_category.src.data_helpers import load_data_and_labels
def train():
    with tf.device('/cpu:0'):
        x_text, y = load_data_and_labels(FLAGS.train_path)

        vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(FLAGS.max_sentence_length)
        x = np.array(list(vocab_processor.fit_transform(x_text)))
        np.random.seed(10)
        shuffle_indices = np.random.permutation(np.arange(len(y)))

        x_shuffled= x[shuffle_indices]
        y_shuffled= y[shuffle_indices]

        dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
        x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
        y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
        print("Train/Dev split: {:d}/{:d}\n".format(len(y_train), len(y_dev)))

        with tf.Graph().as_default():
            session_conf = tf.ConfigProto(
                allow_soft_placement=FLAGS.allow_soft_placement,
                log_device_placement=FLAGS.log_device_placement
            )
            session_conf.gpu_options.allow_growth = FLAGS.gpu_allow_growth
            sess = tf.Session(config=session_conf)
            with sess.as_default():
                model = AttLSTM(
                    sentence_length=x_train.shape[1],
                    num_classes=y_train.shape[1],
                    vocab_size=len(vocab_processor.vocabulary_),
                    embedding_size=FLAGS.embedding_dim,
                    hidden_size=FLAGS.hidden_size,
                    l2_reg_lambda=FLAGS.l2_reg_lambda)

                global_step = tf.Variable(0, name='global_step', trainable=False)


def main():
    pass


if __name__ == "__main__":
    main()
