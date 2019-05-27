import numpy as np
import collections
import random
import pickle
from convai2 import dts_ConvAI2
from extraction import KeywordExtractor
from data_utils import *

class dts_Target(dts_ConvAI2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_vocab(self):
        counter = collections.Counter()
        dialogs = self.get_dialogs()
        for dialog in dialogs:
            for uttr in dialog:
                counter.update(simp_tokenize(uttr))
        print('total vocab count: ', len(counter.items()))
        vocab = [token for token, times in sorted(list(counter.items()), key=lambda x: (-x[1], x[0]))]
        with open('../tx_data/vocab.txt','w') as f:
            for word in vocab:
                f.write(word + '\n')
        print('save vocab in vocab.txt')
        return vocab

    def get_kwsess(self, vocab, mode='all'):
        keyword_extractor = KeywordExtractor(vocab)
        corpus = self.get_data(mode = mode, cands=False)
        sess_set = []
        for sess in corpus:
            data = {}
            data['history'] = ''
            data['dialog'] = []
            for dialog in sess['dialog']:
                data['dialog'].append(dialog)
                data['history'] = data['history'] + ' ' + dialog
            data['kws'] = keyword_extractor.extract(data['history'])
            sess_set.append(data)
        return sess_set

    def cal_idf(self):
        counter = collections.Counter()
        dialogs = self.get_dialogs()
        total = 0.
        for dialog in dialogs:
            for uttr in dialog:
                total += 1
                counter.update(set(kw_tokenize(uttr)))
        idf_dict = {}
        for k,v in counter.items():
            idf_dict[k] = np.log10(total / (v+1.))
        return idf_dict

    def make_dataset(self):
        vocab = self.get_vocab()
        idf_dict = self.cal_idf()
        kw_counter = collections.Counter()
        sess_set = self.get_kwsess(vocab)
        for data in sess_set:
            kw_counter.update(data['kws'])
        kw_freq = {}
        kw_sum = sum(kw_counter.values())
        for k, v in kw_counter.most_common():
            kw_freq[k] = v / kw_sum
        for data in sess_set:
            data['score'] = 0.
            for kw in set(data['kws']):
                data['score'] += kw_freq[kw]
            data['score'] /= len(set(data['kws']))
        sess_set.sort(key=lambda x: x['score'], reverse=True)

        all_data = {'train':[], 'valid':[], 'test':[]}
        keyword_extractor = KeywordExtractor(idf_dict)
        for id, sess in enumerate(sess_set):
            type = 'train'
            if id < 500:
                type = 'test'
            elif random.random() < 0.05:
                type = 'valid'
            sample = {'dialog':sess['dialog'], 'kwlist':[]}
            for i in range(len(sess['dialog'])):
                sample['kwlist'].append(keyword_extractor.idf_extract(sess['dialog'][i]))
            all_data[type].append(sample)
        pickle.dump(all_data, open('source_data.pk','wb'))
        return all_data