#!/usr/bin/python
#coding:utf-8
import tensorflow as tf


# def attention(inputs):
#     # Trainable parameters
#     print(inputs.shape)
#
#     hidden_size = inputs.shape[2].value
#     u_omega = tf.get_variable("u_omega", [hidden_size], initializer=tf.keras.initializers.glorot_normal())
#
#     with tf.name_scope('v'):
#         v = tf.tanh(inputs)
#
#     # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
#     vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
#     vu_shape = tf.shape(vu)
#     alphas = tf.nn.softmax(vu, name='alphas')  # (B,T) shape
#
#     # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
#     output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)
#
#     # Final output with tanh
#     output = tf.tanh(output)
#
#     output_shape = tf.shape(output)
#     alphas_shape = tf.shape(alphas)
#
#     return output, alphas


def attention(inputs, attention_size, time_major=False, return_alphas=False):


    if isinstance(inputs, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        inputs = tf.concat(inputs, 2)

    if time_major:
        # (T,B,D) => (B,T,D)
        inputs = tf.array_ops.transpose(inputs, [1, 0, 2])

    hidden_size = inputs.shape[2].value  # D value - hidden size of the RNN layer

    # Trainable parameters
    w_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
    b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
    u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

    with tf.name_scope('v'):
        # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
        #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
        v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)

    # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
    vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
    alphas = tf.nn.softmax(vu, name='alphas')         # (B,T) shape

    # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
    output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)

    output_shape = tf.shape(output)
    if not return_alphas:
        return output
    else:
        return output, alphas

