# coding : utf-8

import numpy as np
import tensorflow as tf


class TextCNN(object):
    def __init__(self,
                 filter_sizes,
                 num_filters,
                 num_classes,
                 learning_rate,
                 batch_size,
                 decay_rate,
                 decay_steps,
                 sequence_length,
                 vocab_size,
                 embed_size,
                 is_training,
                 initializer=tf.random_normal_initializer(stddev=0.1),
                 multi_label_flag=False,
                 clip_gradients=5.0,
                 decay_rate_big=0.5):
        """Init all hyperparameter."""
        self.filter_sizes = filter_sizes  # e.g. [3,4,5]
        self.num_filters = num_filters
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.is_training = is_training
        self.initializer = initializer

        self.num_filters_total = self.num_filters * len(filter_sizes)
        self.multi_label_flag = multi_label_flag
        self.clip_gradients = clip_gradients

        self.learning_rate = tf.Variable(learning_rate, trainable=False, name="learning_rate")
        self.learning_rate_decay_half_op = tf.assign(self.learning_rate, self.learning_rate*decay_rate_big)

        # add placeholder (X, label)
        self.input_x = tf.placeholder(
            tf.int32, [None, self.sentence_len], name="input_x")
        self.input_y = tf.placeholder(tf.int32, [None], name="input_y")
        # only for multilabel
        self.input_y_multilabel = tf.placeholder(tf.float32, [None, self.num_classes], name="input_y_multilabel")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        self.epoch_step = tf.Variable(0, trainable=False, name="epoch_step")
        self.epoch_increment = tf.assign(self.epoch_step,
                                         tf.add(self.epoch_step,
                                                tf.constant(1)))

        self.decay_steps, self.decay_rate = decay_steps, decay_rate
        # Init embedding matrix and W, b
        self.instantiate_weights()
        # [None, self.label_size]
        self.logits = self.inference()
        if not is_training:
            return
        if multi_label_flag:
            print("going to use multi labels loss.")
            self.loss_val = self.loss_multilabel()
        else:
            print("going to use single label loss.")
            self.loss_val = self.loss()

        self.train_op = self.train()
        # [None,]
        self.predictions = tf.argmax(self.logits, axis=1, name="predictions")

        if not self.multi_label_flag:
            # tf.argmax(self.logits, 1) --> [batch_size]
            correct_prediction = tf.equal(
                tf.cast(self.predictions, tf.int32), self.input_y)
            self.accuracy = tf.reduce_mean(
                tf.cast(correct_prediction, tf.float32), name="accuracy")
        else:
            self.accuracy = tf.constant(0.5)  # TODO fake acccuracy

    def instantiate_weights(self):
        """Define all weights here."""
        with tf.name_scope("embedding"):
            # embedding matrix
            self.embedding = tf.get_variable("embedding",
                                             [self.vocab_size, self.embed_size], initializer=self.initializer)
            self.W = tf.get_variable("W", [self.num_filters_total, self.num_classes])
            self.b = tf.get_variable("b", [self.num_classes])

    def inference(self):
        """
        Main computation graph here:
        1. embedding
        2. average
        3. linear classfier
        """
        # 1. get embedding of words in the sentence
        # [None, sentence_len, embed_size]
        sentence_embeddings = tf.nn.embedding_lookup(self.embedding,
                                                     self.input_x)

        # 2. average vectors, to get representation of the sentence
        # [None, sentence_len, embed_size,1]
        self.sentence_embeddings = tf.expand_dims(sentence_embeddings, -1)

        # 3. conv layer
        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            # create filter
            _filter = tf.get_variable("filter-%s" % filter_size, [filter_size, self.embed_size, 1, self.num_filters], initializer=self.initializer)
            # conv operation
            conv = tf.nn.conv2d(self.sentence_embeddings, _filter, strides=[1, 1, 1, 1], padding="VALID", name="conv")
            # apply nolinearity
            b = tf.get_variable("b-%s" % filter_size, [self.num_filters])
            # [batch_size, sequence_length-filter_size+1, 1, num_filters]
            h = tf.nn.relu(tf.nn.bias_add(conv, b), "relu")
            # [batch_size, 1, 1, num_filters]
            pooled = tf.nn.max_pool(h, ksize=[1, self.sequence_length-filter_size+1, 1, 1], strides=[1, 1, 1, 1], padding="VALID", name="pool")
            pooled_outputs.append(pooled)

        # combine all pooled feature
        # [batch_size, 1, 1, num_filters_total]
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, self.num_filters_total])

        # add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, keep_prob=self.dropout_keep_prob)

        # logits and predictions
        with tf.name_scope("output"):
            logits = tf.matmul(self.h_drop, self.W) + self.b

        return logits

    def loss(self, l2_lambda=1e-4):
        """Calculate loss using (NCE)cross entropy here."""
        # Compute the average NCE loss for the batch.
        # tf.nce_loss automatically draws a new sample of the negative labels each time we evalute the loss
        with tf.name_scope("loss"):
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.logits)
            print("sparse_softmax_cross_entropy_with_logits.losses: ", losses)
            loss = tf.reduce_mean(losses)

            l2_loss = tf.add_n([
                tf.nn.l2_loss(v) for v in tf.trainable_variables()
                if 'bias' not in v.name
            ]) * l2_lambda
            loss += l2_loss

        return loss

    def loss_multilabel(self, l2_lambda=1e-5):
        with tf.name_scope("loss"):
            # let x=logits, z=labels, loss is : z* -log(sigmoid(x)) + (1-z) * -log(1-sigmoid(x))
            losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_y_multilabel, logits=self.logits)
            print("sigmoid_cross_entropy_with_logits.losses:", losses)
            # loss for all data in the batch
            losses = tf.reduce_sum(losses, axis=1)
            loss = tf.reduce_mean(losses)
            l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
            loss += l2_losses
            return loss

    def train(self):
        """Based on the loss, use Adam to update parameter."""
        learning_rate = tf.train.exponential_decay(
            self.learning_rate,
            self.global_step,
            self.decay_steps,
            self.decay_rate,
            staircase=True)

        train_op = tf.contrib.layers.optimize_loss(
            self.loss_val,
            global_step=self.global_step,
            learning_rate=learning_rate,
            optimizer='Adam')
        return train_op


# def test():
#     """
#     Below is a function test; if you use this for text classifiction,
#     you need to tranform sentence to indices of vocabulary first.
#     then feed data to the graph.
#     """
#     num_classes = 19
#     learning_rate = 0.01
#     batch_size = 8
#     decay_steps = 1000
#     decay_rate = 0.9
#     sequence_length = 5
#     vocab_size = 10000
#     embed_size = 100
#     is_training = True
#     # dropout_keep_prob = 1
#
#     fastText = FastText(num_classes, learning_rate, batch_size, decay_steps,
#                         decay_rate, 5, sequence_length, vocab_size, embed_size,
#                         is_training)
#
#     with tf.Session() as sess:
#         sess.run(tf.global_variable_initializer())
#         for i in range(1000):
#             input_x = np.zeros((batch_size, sequence_length), dtype=np.int32)
#             input_y = np.array([1, 0, 1, 1, 1, 2, 1, 1], dtype=np.int32)
#             loss, acc, predict, _ = sess.run(
#                 [
#                     fastText.loss_val, fastText.accuracy, fastText.predictions,
#                     fastText.train_op
#                 ],
#                 feed_dict={
#                     fastText.sequence_length: input_x,
#                     fastText.labels: input_y
#                 })
#             print("loss:", loss, "acc:", acc, "label:", input_y,
#                   "predictions:", predict)
