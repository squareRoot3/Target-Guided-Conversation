import tensorflow as tf
import importlib
import os
if __name__ == '__main__':
    flags = tf.flags
    flags.DEFINE_string('data', 'data_config', 'The data config')
    flags.DEFINE_string('agent', 'kernel', 'The predictor type')
    flags.DEFINE_string('mode', 'train', 'The mode')

    FLAGS = flags.FLAGS
    config_data = importlib.import_module('config.' + FLAGS.data)
    config_model = importlib.import_module('config.' + FLAGS.agent)
    model = importlib.import_module('model.' + FLAGS.agent)
    predictor = model.Predictor(config_model, config_data, FLAGS.mode)
    if not os.path.exists('save/'+FLAGS.agent):
        os.makedirs('save/'+FLAGS.agent)

    if FLAGS.mode == 'train_kw':
        predictor.train_keywords()
    if FLAGS.mode == 'test_kw':
        predictor.test_keywords()
    if FLAGS.mode == 'train':
        predictor.train()
        predictor.test()
    if FLAGS.mode == 'test':
        predictor.test()
