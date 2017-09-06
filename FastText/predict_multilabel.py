# coding:utf-8

import codecs
import os
from config import params as FLAGS

import numpy as np
import tensorflow as tf
from data_util_zhihu import (create_voabulary, create_voabulary_label,
                             load_data_predict, load_final_test_data)
from model import FastText
from tflearn.data_utils import pad_sequences


def main():
    # 1.load data with vocabulary of words and label_size
    vocabulary_word2index, vocabulary_index2word = create_voabulary()
    vocabulary_size = len(vocabulary_index2word)
    vocabulary_word2index_label, vocabulary_index2word_label = create_voabulary_label(
    )
    questionId_question_lists = load_final_test_data(FLAGS.predict_source_file)  # TODO
    test = load_data_predict(vocabulary_word2index, vocabulary_word2index,
                             questionId_question_lists)  # TODO
    testX = []
    question_id_list = []
    for tup in test:
        question_id, question_string_list = tup
        question_id_list.append(question_id)
        testX.append(question_string_list)

    # 2. Data preprocessing: Sequence padding
    print("start padding....")
    testX2 = pad_sequences(testX, maxlen=FLAGS.sentence_size, value=0.)
    print("end padding....")

    # 3.creat session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # instantiate model
        fastText = FastText(
            label_size=FLAGS.label_size,
            learning_rate=FLAGS.learning_rate,
            batch_size=FLAGS.batch_size,
            decay_rate=FLAGS.decay_rate,
            decay_steps=FLAGS.decay_steps,
            num_sampled=FLAGS.num_sampled,
            vocab_size=FLAGS.vocab_size,
            embed_size=FLAGS.embed_size,
            is_training=FLAGS.is_training)
        saver = tf.train.Saver()
        if os.path.exists(FLAGS.ckpt_dir + "checkpoint"):
            print("Restoring Variables from checkpoint")
            saver.restore(sess, tf.train.latest_checkpoint(FLAGS.ckpt_dir))
        else:
            print("can't find the checkpoint. Going to stop")
            return
        # 5. feed data, to get logits
        number_of_trainding_data = len(testX2)
        print("num of trainding data:", number_of_trainding_data)
        batch_size = 1
        index = 0
        predict_target_file_f = codecs.open(FLAGS.predict_target_file, 'a',
                                            'utf-8')

        for start, end in zip(
                range(0, number_of_trainding_data, batch_size),
                range(batch_size, number_of_trainding_data, batch_size)):
            logits = sess.run(
                fastText.logits,
                feed_dict={
                    fastText.sentence: testX2[start:end],
                    fastText.labels: testX2[start:end]
                })
            # 6. get label using logits
            predicted_labels = get_label_using_logits(
                logits[0], vocabulary_index2word_label)
            # 7. write question id and labels to file system
            write_question_id_with_labels(question_id_list[index],
                                          predicted_labels,
                                          predict_target_file_f)
            index += 1
        predict_target_file_f.close()


def get_label_using_logits(logits, vocabulary_index2word_label, top_numer=5):
    index_list = np.argsort(logits)[-top_numer:]
    index_list = index_list[::-1]
    label_list = []
    for index in index_list:
        label = vocabulary_index2word_label[index]
        label_list.append(label)
    return label_list


def write_question_id_with_labels(question_id, label_list, f):
    labels_string = ",".join(label_list)
    f.write(question_id+","+labels_string)


if __name__ == '__main__':
    tf.app.run()
