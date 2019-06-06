import texar as tx
import tensorflow as tf
import numpy as np


class Predictor():
    def __init__(self, config_model, config_data, mode=None):
        self.config = config_model
        self.data_config = config_data
        self.build_model()
        self.gpu_config = tf.ConfigProto()
        self.gpu_config.gpu_options.allow_growth = True

    def build_model(self):
        self.train_data = tx.data.MultiAlignedData(self.data_config.data_hparams['train'])
        self.valid_data = tx.data.MultiAlignedData(self.data_config.data_hparams['valid'])
        self.test_data = tx.data.MultiAlignedData(self.data_config.data_hparams['test'])
        self.iterator = tx.data.TrainTestDataIterator(train=self.train_data, val=self.valid_data, test=self.test_data)
        self.vocab = self.train_data.vocab(0)
        self.embedder = tx.modules.WordEmbedder(init_value=self.train_data.embedding_init_value(0).word_vecs)
        self.source_encoder = tx.modules.HierarchicalRNNEncoder(hparams=self.config.source_encoder_hparams)
        self.target_encoder = tx.modules.UnidirectionalRNNEncoder(hparams=self.config.target_encoder_hparams)
        self.linear_matcher = tx.modules.MLPTransformConnector(1)

    def forward(self, batch):
        source_embed = self.embedder(batch['source_text_ids'])
        target_embed = self.embedder(batch['target_text_ids'])
        target_embed = tf.reshape(target_embed, [-1, self.data_config._max_seq_len + 2, self.embedder.dim])
        source_code = self.source_encoder(source_embed,
                                          sequence_length_minor=batch['source_length'],
                                          sequence_length_major=batch['source_utterance_cnt'])[1]
        target_length = tf.reshape(batch['target_length'], [-1])
        target_code = self.target_encoder(target_embed, sequence_length=target_length)[1]
        target_code = tf.reshape(target_code, [-1, 20, self.config._code_len])
        source_code = tf.expand_dims(source_code, 1)
        source_code = tf.tile(source_code, [1, 20, 1])
        feature_code = target_code * source_code
        feature_code = tf.reshape(feature_code, [-1, self.config._code_len])
        logits = self.linear_matcher(feature_code)
        logits = tf.reshape(logits, [-1, 20])
        labels = tf.one_hot(batch['label'], 20)
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits))
        ans = tf.arg_max(logits, -1)
        acc = tx.evals.accuracy(batch['label'], ans)
        rank = tf.nn.top_k(logits, k=20)[1]
        return loss, acc, rank

    def train(self):
        batch = self.iterator.get_next()
        loss, acc, _ = self.forward(batch)
        op_step = tf.Variable(0, name='op_step')
        train_op = tx.core.get_train_op(loss, global_step=op_step, hparams=self.config.opt_hparams)
        max_val_acc = 0.
        self.saver = tf.train.Saver()
        with tf.Session(config=self.gpu_config) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            sess.run(tf.tables_initializer())
            for epoch_id in range(self.config._max_epoch):
                self.iterator.switch_to_train_data(sess)
                cur_step = 0
                cnt_acc = []
                while True:
                    try:
                        cur_step += 1
                        feed = {tx.global_mode(): tf.estimator.ModeKeys.TRAIN}
                        loss, acc_ = sess.run([train_op, acc], feed_dict=feed)
                        cnt_acc.append(acc_)
                        if cur_step % 200 == 0:
                            print('batch {}, loss={}, acc1={}'.format(cur_step, loss, np.mean(cnt_acc[-200:])))
                    except tf.errors.OutOfRangeError:
                        break
                op_step = op_step + 1
                self.iterator.switch_to_val_data(sess)
                cnt_acc = []
                while True:
                    try:
                        feed = {tx.global_mode(): tf.estimator.ModeKeys.EVAL}
                        acc_ = sess.run([acc], feed_dict=feed)
                        cnt_acc.append(acc_)
                    except tf.errors.OutOfRangeError:
                        mean_acc = np.mean(cnt_acc)
                        print('valid acc1={}'.format(mean_acc))
                        if mean_acc > max_val_acc:
                            max_val_acc = mean_acc
                            self.saver.save(sess, self.config._save_path)
                        break

    def test(self):
        batch = self.iterator.get_next()
        loss, acc, rank = self.forward(batch)
        with tf.Session(config=self.gpu_config) as sess:
            sess.run(tf.tables_initializer())
            self.saver = tf.train.Saver()
            self.saver.restore(sess, self.config._save_path)
            self.iterator.switch_to_test_data(sess)
            rank_cnt = []
            while True:
                try:
                    feed = {tx.global_mode(): tf.estimator.ModeKeys.PREDICT}
                    ranks, labels = sess.run([rank, batch['label']], feed_dict=feed)
                    for i in range(len(ranks)):
                        rank_cnt.append(np.where(ranks[i]==labels[i])[0][0])
                except tf.errors.OutOfRangeError:
                    rec = [0,0,0,0,0]
                    MRR = 0
                    for rank in rank_cnt:
                        for i in range(5):
                            rec[i] += (rank <= i)
                        MRR += 1 / (rank+1)
                    print('test rec1@20={:.4f}, rec3@20={:.4f}, rec5@20={:.4f}, MRR={:.4f}'.format(
                        rec[0]/len(rank_cnt), rec[2]/len(rank_cnt), rec[4]/len(rank_cnt), MRR/len(rank_cnt)))
                    break

    def retrieve_init(self, sess):
        data_batch = self.iterator.get_next()
        loss, acc, _ = self.forward(data_batch)
        self.corpus = self.data_config._corpus
        self.corpus_data = tx.data.MonoTextData(self.data_config.corpus_hparams)
        corpus_iterator = tx.data.DataIterator(self.corpus_data)
        batch = corpus_iterator.get_next()
        corpus_embed = self.embedder(batch['corpus_text_ids'])
        utter_code = self.target_encoder(corpus_embed, sequence_length=batch['corpus_length'])[1]
        self.corpus_code = np.zeros([0, self.config._code_len])
        corpus_iterator.switch_to_dataset(sess)
        sess.run(tf.tables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, self.config._save_path)
        feed = {tx.global_mode(): tf.estimator.ModeKeys.PREDICT}
        while True:
            try:
                utter_code_ = sess.run(utter_code, feed_dict=feed)
                self.corpus_code = np.concatenate([self.corpus_code, utter_code_], axis=0)
            except tf.errors.OutOfRangeError:
                break

        self.minor_length_input = tf.placeholder(dtype=tf.int32, shape=(1, 9))
        self.major_length_input = tf.placeholder(dtype=tf.int32, shape=(1))
        self.history_input = tf.placeholder(dtype=object, shape=(9, self.data_config._max_seq_len + 2))

        history_ids = self.vocab.map_tokens_to_ids(self.history_input)
        history_embed = self.embedder(history_ids)
        history_code = self.source_encoder(tf.expand_dims(history_embed, axis=0),
                                           sequence_length_minor=self.minor_length_input,
                                           sequence_length_major=self.major_length_input)[1]
        select_corpus = tf.cast(self.corpus_code, dtype=tf.float32)
        feature_code = self.linear_matcher(select_corpus * history_code)
        self.ans_output = tf.nn.top_k(tf.squeeze(feature_code, 1), k=self.data_config._retrieval_candidates)[1]

    def retrieve(self, source, sess):
        history, seq_len, turns, context, context_len = source
        ans = sess.run(self.ans_output, feed_dict={self.history_input: history,
                                                   self.minor_length_input: [seq_len],
                                                   self.major_length_input: [turns]})
        for i in range(self.data_config._max_turns + 1):
            if ans[i] not in self.reply_list:  # avoid repeat
                self.reply_list.append(ans[i])
                reply = self.corpus[ans[i]]
                break
        return reply
