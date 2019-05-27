import tensorflow as tf
import importlib
import random
from preprocess.data_utils import utter_preprocess, is_reach_goal
from model import retrieval

class Target_Simulation():
    def __init__(self, config_model, config_data, config_retrieval):
        g1 = tf.Graph()
        with g1.as_default():
            self.retrieval_agent = retrieval.Predictor(config_retrieval, config_data)
            sess1 = tf.Session(graph=g1, config=self.retrieval_agent.gpu_config)
            self.retrieval_agent.retrieve_init(sess1)
        g2 = tf.Graph()
        with g2.as_default():
            self.target_agent = model.Predictor(config_model, config_data)
            sess2 = tf.Session(graph=g2, config=self.target_agent.gpu_config)
            self.target_agent.retrieve_init(sess2)
        self.start_utter = config_data._start_corpus
        success_cnt, turns_cnt = 0, 0
        for i in range(int(FLAGS.times)):
            print('--------Session {} --------'.format(i))
            success, turns = self.simulate(sess1, sess2)
            success_cnt += success
            turns_cnt += turns
        print('success time {}, average turns {:.2f}'.format(success_cnt, turns_cnt / success_cnt))

    def simulate(self, sess1, sess2):
        history = []
        history.append(random.sample(self.start_utter,1)[0])
        target_kw = random.sample(target_set,1)[0]
        self.target_agent.target = target_kw
        self.target_agent.score = 0.
        self.target_agent.reply_list = []
        self.retrieval_agent.reply_list = []

        print('START: ' + history[0])
        for i in range(config_data._max_turns):
            source = utter_preprocess(history, config_data._max_seq_len)
            reply = self.retrieval_agent.retrieve(source, sess1)
            print('retrieval_agent: ', reply)
            history.append(reply)
            source = utter_preprocess(history, config_data._max_seq_len)
            reply = self.target_agent.retrieve(source, sess2)
            print('{}_agent: '.format(FLAGS.agent), reply)
            print('Keyword: {}, Similarity: {:.2f}'.format(self.target_agent.next_kw, self.target_agent.score))
            history.append(reply)
            if is_reach_goal(history[-2] + history[-1], target_kw):
                print('Successfully chat to the target \'{}\'.'.format(target_kw))
                return (True, (len(history)+1)//2)

        print('Failed by reaching the maximum turn, target: \'{}\'.'.format(target_kw))
        return (False, 0)

if __name__ == '__main__':
    flags = tf.flags
    flags.DEFINE_string('agent', 'kernel', 'The agent type, supports kernel / matrix / neural / retrieval.')
    flags.DEFINE_string('times', '100', 'Simulation times.')

    FLAGS = flags.FLAGS
    config_data = importlib.import_module('config.data_config')
    config_model = importlib.import_module('config.' + FLAGS.agent)
    config_retrieval = importlib.import_module('config.retrieval')
    model = importlib.import_module('model.' + FLAGS.agent)

    target_set = []
    for line in open('tx_data/test/keywords.txt', 'r').readlines():
        target_set = target_set + line.strip().split(' ')

    Target_Simulation(config_model,config_data,config_retrieval)