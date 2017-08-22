import os
import pickle

import numpy as np
import tensorflow as tf
import word2vec
from model import FastText
from p4_zhihu_load_data import (creat_vocabulary, creat_vocabulary_label,
                                load_data)
from tflearn.data_utils import pad_sentences, to_categorical

# configuration
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("label_size", 1999, "number of labels")
tf.app.flags.DEFINE_float("learning_rate", 0.01, "learning rate")
tf.app.flags.DEFINE_integer("batch_size", 128,
                            "batch size for training/evaluting")
tf.app.flags.DEFINE_integer("decay_steps", 20000,
                            "how many steps before decay learning rate")
tf.app.flags.DEFINE_float("decay_rate", 0.8, "rate of decay for learning rate")
tf.app.flags.DEFINE_integer("num_sampled", 50, "number of noise sampling")
tf.app.flags.DEFINE_string("ckpt_dir", "fastText_checkpoints/",
                           "checkpoint location for the model")
tf.app.flags.DEFINE_integer("sentence_size", 200, "max sentence size")
tf.app.flags.DEFINE_integer("embed_size", 100, "embedding size")
tf.app.flags.DEFINE_boolean(
    "is_training", True, "is training.true:training, false:testing/inference")
tf.app.flags.DEFINE_integer("num_epoches", 15, "numbers of training epoch")
tf.app.flags.DEFINE_integer("validate_every", 10,
                            "validate every validate every epoch")
tf.app.flags.DEFINE_boolean("use_embedding", True,
                            "whether to use embedding or not")
tf.app.flags.DEFINE_string("cache_path", "fastText_checkpoints/data_cache.pik",
                           "checkpoint location for the model")


def main():
    """
    1. load data(X: list, Y: int)
    2. creat session
    3. feed data
    4. training
    5. validation
    6. prediction
    """
    trainX, trainY, testX, testY = None, None, None, None
    vocabulary_word2index, vocabulary_index2word = creat_vocabulary()
    vocabulay_size = len(vocabulary_word2index)
    vocabulary_word2index_label, _ = creat_vocabulary_label()
    train, test, _ = load_data(
        vocabulary_word2index, vocabulary_word2index_label, data_type="train")
    trainX, trainY = train
    testX, testY = test

    print("testX.shape: ", np.array(testX.shape))
    print("testY shape: ", np.array(testY.shape))
    print("testX[0]: ", testX[0])
    print("testY[0]: ", testY[0])

    print("starting padding & tranform to one hot....")
    trainX = pad_sentences(trainX, maxlen=FLAGS.sentence_len, value=0.)
    testX = pad_sentences(testX, maxlen=FLAGS.sentence_len, value=0.)
    print("testX[0]: ", testX[0])
    print("testY[0]: ", testY[0])

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        fastText = FastText(
            label_size=FLAGS.label_size,
            learning_rate=FLAGS.learning_rate,
            batch_size=FLAGS.batch_size,
            decay_steps=FLAGS.decay_steps,
            decay_rate=FLAGS.decay_rate,
            num_sampled=FLAGS.num_sampled,
            sentence_len=FLAGS.sentence_len,
            embed_size=FLAGS.embed_size,
            vocab_size=vocabulay_size,
            is_training=FLAGS.is_training)

        saver = tf.train.saver()
        if os.path.exists(FLAGS.ckpt_dir + "checkpoint"):
            print("Restoring variable from checkpoint.")
            saver.restore(sess, tf.train.lastest_checkpoint(FLAGS.ckpt_dir))
        else:
            print("Initializing variables.")
            sess.run(tf.global_variables_initializer())
            if FLAGS.use_embedding:
                assign_pretrained_word_embedding(sess, vocabulary_index2word,
                                                 vocabulay_size, fastText)

        current_epoch = sess.run(fastText.epoch_step)

        # feed data to training
        number_of_training_data = len(trainX)
        batch_size = FLAGS.batch_size
        for epoch in range(current_epoch, FLAGS.num_epoches):
            loss, acc, counter = 0., 0., 0
            for start, end in zip(
                    range(0, number_of_training_data, batch_size),
                    range(batch_size, number_of_training_data, batch_size)):
                if epoch_step == 0 and counter == 0:
                    print("trainX[start:end]: ", trainX[start:end])
                    print("trainY[start:end]: ", trainY[start:end])
                curr_loss, curr_acc, _ = sess.run(
                    [fastText.loss_val, fastText.accuracy, fastText.train_op],
                    feed_dict={
                        fastText.sentence: trainX[start:end],
                        fastText.labels: trainY[start:end]
                    })
                loss, acc, counter = loss+curr_loss, acc+curr_acc, counter+1
