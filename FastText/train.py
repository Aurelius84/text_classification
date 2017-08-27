import os

import numpy as np
import tensorflow as tf
import word2vec
from model import FastText
from p4_zhihu_load_data import (creat_vocabulary, creat_vocabulary_label,
                                load_data)
from tflearn.data_utils import pad_sentences
from config import params as FLAGS


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
                if epoch == 0 and counter == 0:
                    print("trainX[start:end]: ", trainX[start:end])
                    print("trainY[start:end]: ", trainY[start:end])
                curr_loss, curr_acc, _ = sess.run(
                    [fastText.loss_val, fastText.accuracy, fastText.train_op],
                    feed_dict={
                        fastText.sentence: trainX[start:end],
                        fastText.labels: trainY[start:end]
                    })
                loss, acc, counter = loss + curr_loss, acc + curr_acc, counter + 1
                if counter % 500 == 0:
                    print(
                        "Epoch %d\tBatch %d\tTrain Loass: %.3f\tTrain Acc:%.3f"
                        % (epoch, counter, loss / float(counter),
                           acc / float(counter)))
            # epoch increment
            print("going to increment epoch counter....")
            sess.run(fastText.epoch_increment)

            # 4. validation
            print(epoch, FLAGS.validate_every,
                  (epoch % FLAGS.validate_every == 0))
            if epoch % FLAGS.validate_every == 0:
                eval_loss, eval_acc = do_eval(sess, fastText, testX, testY,
                                              batch_size)
                print("epoch %d\tvalidation Loss:%.3f\tvalidation Acc:%.3f" %
                      (epoch, eval_loss, eval_acc))

                # save model to checkpoint
                save_path = FLAGS.ckpt_dir + "model.ckpt"
                saver.save(sess, save_path, global_step=fastText.epoch_step)

        # 5.最后在测试集上做测试，并报告测试准确率
        test_loss, test_acc = do_eval(sess, fastText, testX, testY, batch_size)


def assign_pretrained_word_embedding(sess, vocabulary_index2word,
                                     vocabulay_size, fastText):
    print("using pre-trained word embedding.started....")
    word2vec_model = word2vec.load(
        'zhihu-word2vec-multilabel.bin-100', kind='bin')
    word2vec_dict = {}
    for word, vector in zip(word2vec_model.vocab, word2vec_model.vectors):
        word2vec_dict[word] = vector
    word_embedding_2dlist = [[]] * vocabulay_size
    # assign empty for first word: 'PAD'
    word_embedding_2dlist[0] = np.zeros(FLAGS.embed_size)
    # bound for random variables
    bound = np.sqrt(6.0) / np.sqrt(vocabulay_size)
    count_exist = 0
    count_not_exist = 0
    for i in range(1, vocabulay_size):
        word = vocabulary_index2word[i]
        embedding = None
        try:
            embedding = word2vec_dict[word]
        except Exception:
            embedding = None
        if embedding is not None:
            word_embedding_2dlist[i] = embedding
            count_exist += 1
        else:
            word_embedding_2dlist[i] = np.random.uniform(
                -bound, bound, FLAGS.embed_size)
            count_not_exist += 1
    word_embedding_final = np.array(word_embedding_2dlist)
    word_embedding = tf.constant(word_embedding_final, dtype=tf.float32)
    t_assign_embedding = tf.assign(fastText.embedding, word_embedding)
    sess.run(t_assign_embedding)
    print("word, exists embedding:", count_exist,
          " ;word not exist embedding:", count_not_exist)
    print("using pre-trained word embedding.ended...")


def do_eval(sess, fastText, evalX, evalY, batch_size):
    number_example = len(evalX)
    eval_loss, eval_acc, eval_counter = 0., 0., 0
    for start, end in zip(
            range(0, number_example, batch_size),
            range(batch_size, number_example, batch_size)):
        curr_eval_loss, curr_eval_acc = sess.run(
            [fastText.loss_val, fastText.accuracy],
            feed_dict={
                fastText.sentence: evalX[start:end],
                fastText.labels: evalY[start:end]
            })

        eval_loss += curr_eval_loss
        eval_acc += curr_eval_acc
        eval_counter += 1

        return eval_loss / eval_counter, eval_acc / eval_counter


if __name__ == '__main__':
    tf.app.run()
