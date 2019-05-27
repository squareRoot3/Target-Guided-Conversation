from dataset import dts_Target
from collections import Counter
import pickle
import random
import os
import shutil
if not os.path.exists('../tx_data'):
    os.mkdir('../tx_data')
    os.mkdir('../tx_data/train')
    os.mkdir('../tx_data/valid')
    os.mkdir('../tx_data/test')

# import texar
# if not os.path.exists('convai2/source'):
#     print('Downloading source ConvAI2 data')
#     texar.data.maybe_download('https://drive.google.com/file/d/1LPxNIVO52hZOwbV3Zply_ITi2Uacit-V/view?usp=sharing'
#                                 ,'convai2', extract=True)

shutil.copy('convai2/source/embedding.txt', '../tx_data/embedding.txt')
dataset = dts_Target()
dataset.make_dataset()

data = pickle.load(open("source_data.pk","rb"))
max_utter = 9
candidate_num = 20
start_corpus_file = open("../tx_data/start_corpus.txt", "w")
corpus_file = open("../tx_data/corpus.txt", "w")

for stage in ['train', 'valid', 'test']:
    source_file = open("../tx_data/{}/source.txt".format(stage), "w")
    target_file = open("../tx_data/{}/target.txt".format(stage), "w")
    context_file = open("../tx_data/{}/context.txt".format(stage), "w")
    keywords_file = open("../tx_data/{}/keywords.txt".format(stage), "w")
    label_file = open("../tx_data/{}/label.txt".format(stage), "w")
    keywords_vocab_file = open("../tx_data/{}/keywords_vocab.txt".format(stage), "w")
    corpus = []
    keywords_counter = Counter()
    for sample in data[stage]:
        corpus += sample['dialog'][1:]
        start_corpus_file.write(sample['dialog'][0]+ '\n')
        for kws in sample['kwlist']:
            keywords_counter.update(kws)
    for kw, _ in keywords_counter.most_common():
        keywords_vocab_file.write(kw + '\n')
    for sample in data[stage]:
        for i in range(2, len(sample['dialog'])):
            if len(sample['kwlist'][i]) > 0:
                source_list = sample['dialog'][max(0, i - max_utter):i]
                source_str = '|||'.join(source_list)
                while True:
                    random_corpus = random.sample(corpus, candidate_num - 1)
                    if sample['dialog'][i] not in random_corpus:
                        break
                corpus_file.write(sample['dialog'][i] + '\n')
                target_list = [sample['dialog'][i]] + random_corpus
                target_str = '|||'.join(target_list)
                source_file.write(source_str + '\n')
                target_file.write(target_str + '\n')
                context_file.write(' '.join(sample['kwlist'][i-2] +
                    sample['kwlist'][i-1]) + '\n')
                keywords_file.write(' '.join(sample['kwlist'][i]) + '\n')
                label_file.write('0\n')
                
    source_file.close()
    target_file.close()
    label_file.close()
    keywords_vocab_file.close()
    context_file.close()

start_corpus_file.close()
corpus_file.close()
