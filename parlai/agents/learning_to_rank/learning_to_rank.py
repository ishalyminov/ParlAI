import random
import copy

import numpy as np
import torch
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

from alana_learning_to_rank.learning_to_rank import create_model_personachat
from alana_learning_to_rank.util.training_utils import get_loss_function

from parlai.core.agents import Agent
from parlai.core.dict import DictionaryAgent


class LearningToRankAgent(Agent):
    def __init__(self, opt, shared=None):
        # initialize defaults first
        super().__init__(opt, shared)

        self.id = 'LearningToRank'

        if shared:
            raise NotImplementedError
        else:
            # this is not a shared instance of this class, so do full
            # initialization. if shared is set, only set up shared members.
            self.sess = tf.Session()
            self.dict = DictionaryAgent(opt)
            self.EOS = self.dict.end_token
            self.observation = {'text': self.EOS, 'episode_done': True}
            self.learning_to_rank_config = {'max_context_turns': 10,
                                            'max_sequence_length': 60,
                                            'embedding_size': 256,
                                            'vocab_size': len(self.dict),
                                            'rnn_cell': 'GRUCell',
                                            'dropout_prob': 0.3,
                                            'mlp_sizes': [16],
                                            'l2_coef': 1e-5,
                                            'lr': 0.0001,
                                            'optimizer': 'AdamOptimizer',
                                            'answer_candidates_number': 20}

            self.X, self.pred, self.y = create_model_personachat(**(self.learning_to_rank_config))
            self.batch_sample_weight = tf.placeholder(tf.float32, [None, 1], name='sample_weight')
            # Define loss and optimizer
            self.loss_op = get_loss_function(self.pred,
                                             self.y,
                                             self.batch_sample_weight,
                                             l2_coef=self.learning_to_rank_config['l2_coef'])

            self.global_step = tf.Variable(0, trainable=False)
            self.sess.run(tf.assign(self.global_step, 0))
            self.learning_rate = tf.train.cosine_decay(self.learning_to_rank_config['lr'],
                                                       self.global_step,
                                                       2000000,
                                                       alpha=0.001)
            optimizer_class = getattr(tf.train, self.learning_to_rank_config['optimizer'])
            self.optimizer = optimizer_class(learning_rate=self.learning_rate)
            self.train_op = self.optimizer.minimize(self.loss_op, self.global_step)

            self.saver = tf.train.Saver(tf.global_variables())
            self.sess.run(tf.global_variables_initializer())
        self.episode_done = True

    def batchify(self, observations):
        assert len(observations) == 1
        """Convert a list of observations into input & target tensors."""
        def valid(obs):
            # check if this is an example our model should actually process
            return 'text' in obs and len(obs['text']) > 0
        # valid examples and their indices
        try:
            valid_inds, exs = zip(*[(i, ex) for i, ex in
                                    enumerate(observations) if valid(ex)])
        except ValueError:
            # zero examples to process in this batch, so zip failed to unpack
            return None, None, None

        context_turns = [[] for _ in range(self.learning_to_rank_config['max_context_turns'])]
        answers = []
        y = None
        max_seq_len = self.learning_to_rank_config['max_sequence_length']
        for observation_i in exs:
            context_turns_i = list(map(self.parse, observation_i['text'].split('\n')))
            context_turns_i = pad_sequences(context_turns_i, maxlen=max_seq_len) 
            context_turns_i = pad_sequences([context_turns_i],
                                            maxlen=len(context_turns),
                                            value=np.zeros(max_seq_len))[0]
            for j, context_turn in enumerate(context_turns_i):
                context_turns[j].append(context_turn)
            answers_i = list(map(self.parse, observation_i['label_candidates']))
            answers_i = pad_sequences(answers_i, maxlen=max_seq_len) 
            answers = answers_i
            label = observation_i.get('labels', [None])[0]
            if label is not None:
                y = np.zeros((len(observation_i['label_candidates']), 1), dtype=np.float32)
                y[observation_i['label_candidates'].index(label)][0] = 1.0

        context_turns_final = [[] for _ in range(self.learning_to_rank_config['max_context_turns'])]
        for answer in answers:
            for i in range(len(context_turns_final)):
                context_turns_final[i].append(context_turns[i][0])
        X = list(map(np.array, context_turns_final + [answers]))
        y = np.array(y) if y is not None else None
        return X, y, valid_inds

    def predict(self, xs, ys=None):
        """Produce a prediction from our model. Update the model using the
        targets if available.
        """
        batchsize = xs[0].shape[0] 

        loss = 0
        feed_dict = {X_i: batch_x_i for X_i, batch_x_i in zip(self.X, xs)}
        sample_weights = np.expand_dims(np.ones(batchsize), -1)

        with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
            if ys is not None:
                with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
                    feed_dict.update({self.y: ys, self.batch_sample_weight: np.ones((batchsize, 1))})
                    batch_pred, _, train_batch_loss = self.sess.run([self.pred, self.train_op, self.loss_op],
                                                                    feed_dict=feed_dict)
            else:
                batch_pred = self.sess.run(self.pred, feed_dict=feed_dict)
        return np.argmax(batch_pred, axis=-2)

    def parse(self, text):
        """Convert string to token indices."""
        return self.dict.txt2vec(text)

    def v2t(self, vec):
        """Convert token indices to string of tokens."""
        return self.dict.vec2txt(vec)

    def hidden_to_idx(self, hidden, drop=False):
        """Converts hidden state vectors into indices into the dictionary."""
        if hidden.size(0) > 1:
            raise RuntimeError('bad dimensions of tensor:', hidden)
        hidden = hidden.squeeze(0)
        scores = self.d2o(hidden)
        if drop:
            scores = self.dropout(scores)
        scores = self.softmax(scores)
        _max_score, idx = scores.max(1)
        return idx, scores

    def observe(self, observation):
        observation = copy.deepcopy(observation)
        if not self.episode_done:
            # if the last example wasn't the end of an episode, then we need to
            # recall what was said in that example
            prev_dialogue = self.observation['text']
            observation['text'] = prev_dialogue + '\n' + observation['text']
            candidates_list = list(observation['label_candidates'])
            random.shuffle(candidates_list)
            observation['label_candidates'] = tuple(candidates_list)
        self.observation = observation
        self.episode_done = observation['episode_done']
        return observation

    def act(self):
        # call batch_act with this batch of one
        return self.batch_act([self.observation])[0]

    def batch_act(self, observations):
        batchsize = len(observations)
        # initialize a table of replies with this agent's id
        batch_reply = [{'id': self.getID()} for _ in range(batchsize)]

        # convert the observations into batches of inputs and targets
        # valid_inds tells us the indices of all valid examples
        # e.g. for input [{}, {'text': 'hello'}, {}, {}], valid_inds is [1]
        # since the other three elements had no 'text' field
        xs, ys, valid_inds = self.batchify(observations)

        if len(xs) == 0:
            # no valid examples, just return the empty responses we set up
            return batch_reply

        # produce predictions either way, but use the targets if available
        predictions = self.predict(xs, ys)

        for i in range(len(predictions)):
            idx = valid_inds[i]
            batch_reply[idx]['text'] = observations[idx]['label_candidates'][predictions[i]]

        return batch_reply

