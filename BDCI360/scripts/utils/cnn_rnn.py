"""
Contruct CNN ->  RNN models using Keras and tensorflow.
Competition url: http://www.datafountain.cn/#/competitions/276/intro
"""
import tensorflow as tf
from keras.layers import Dense


class CNNRNN(object):

    def __init__(self, title_dim, content_dim):
        # titles
        self.titles = tf.placeholder(tf.float32, [None, title_dim])
        # content
        self.content = tf.placeholder(tf.float, [None, content_dim])
