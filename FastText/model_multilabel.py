# coding:utf-8
import tensorflow as tf


class FastTextMultiLabel(object):
    def __init__(self,
                 label_size,
                 learning_rate,
                 batch_size,
                 decay_steps,
                 decay_rate,
                 num_sampled,
                 sentence_len,
                 vocab_size,
                 embed_size,
                 is_training,
                 max_label_per_sample=5):
        # set hyper parameters
        self.label_size = label_size
        self.batch_size = batch_size
        self.num_sampled = num_sampled
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.sentence_len = sentence_len
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.is_training = is_training
        self.max_label_per_sample = max_label_per_sample

        # add placeholder
        self.sentences = tf.placeholder(
            tf.int32, [None, self.sentence_len], name="sentence")
        self.labels = tf.placeholder(
            tf.int32, [None, self.max_label_per_sample], name="labels")
        self.labels_global = tf.placeholder(tf.int32, [None, self.label_size])

        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        self.epoch_step = tf.Variable(0, trainable=False, name="epoch_step")
        self.epoch_increment = tf.assign(self.epoch_step,
                                         tf.add(self.epoch_step,
                                                tf.constant(1)))

        # init weights
        self.instantiate_weights()

        # main graph
        self.logits = self.inference()

        # loss
        self.loss_val = self.loss()

        # update parameters according loss
        self.train_op = self.train()

    def instantiate_weights(self):
        """Init all model weights needed."""
        self.embedding = tf.get_variable("embedding",
                                         [self.vocab_size, self.embed_size])
        self.W = tf.get_variable("W", [self.embed_size, self.label_size])
        self.b = tf.get_variable("b", [self.label_size])

    def inference(self):
        """Main data-flow compute Graph."""
        # Embedding
        sentence_embedding = tf.nn.embedding_lookup(self.embedding,
                                                    self.sentences)

        # average vector of sentences
        self.sentence_embedding = tf.reduce_mean(sentence_embedding, axis=1)

        # liner classifier layer
        logits = tf.matmul(self.sentence_embedding, self.W) + self.b

        return logits

    def loss(self, l2_lambda=1e-4):
        """Calculate loss."""
        if self.is_training:
            loss = tf.reduce_mean(
                tf.nn.nce_loss(
                    weights=tf.transpose(self.W),
                    biases=self.b,
                    labels=self.labels,
                    inputs=self.sentence_embedding,
                    num_sampled=self.num_sampled,
                    num_true=self.max_label_per_sample,
                    num_classes=self.label_size,
                    partition_strategy="div"))
        else:
            labels_multi_hot = self.labels_global
            loss = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=labels_multi_hot, logits=self.logits)
            loss = tf.reduce_sum(loss, axis=1)
        # regularization
        l2_losses = tf.add_n([
            tf.nn.l2_loss(v) for v in tf.trainable_variables()
            if 'bias' not in v.name
        ]) * l2_lambda
        loss += l2_losses
        return loss

    def train(self):
        """Based on the loss, use SGD."""
        learning_rate = tf.train.exponential_decay(
            learning_rate=self.learning_rate,
            global_step=self.global_step,
            decay_steps=self.decay_steps,
            decay_rate=self.decay_rate,
            staircas=True)

        train_op = tf.contrib.layers.optimize_loss(
            self.loss_val,
            global_step=self.global_step,
            learning_rate=learning_rate,
            optimizer="Adam")
        return train_op
