import os
data_root = './tx_data'
_corpus = [x.strip() for x in open('tx_data/corpus.txt', 'r').readlines()]
_start_corpus = [x.strip() for x in open('tx_data/start_corpus.txt', 'r').readlines()]
_max_seq_len = 30
_num_neg = 20
_max_turns = 8
_batch_size = 64
_retrieval_candidates = 1000

data_hparams = {
    stage: {
        "num_epochs": 1,
        "shuffle": stage != 'test',
        "batch_size": _batch_size,
        "datasets": [
            {  # dialogue history
                "variable_utterance": True,
                "max_utterance_cnt": 9,
                "max_seq_length": _max_seq_len,
                "files": [os.path.join(data_root, '{}/source.txt'.format(stage))],
                "vocab_file": os.path.join(data_root, 'vocab.txt'),
                "embedding_init": {
                    "file": os.path.join(data_root, 'embedding.txt'),
                    "dim": 200,
                    "read_fn": "load_glove"
                },
                "data_name": "source"
            },
            {  # candidate response
                "variable_utterance": True,
                "max_utterance_cnt": 20,
                "max_seq_length": _max_seq_len,
                "files": [os.path.join(data_root, '{}/target.txt'.format(stage))],
                "vocab_share_with": 0,
                "embedding_init_share_with" : 0,
                "data_name": "target"
            },
            {  # context (source keywords)
                "files": [os.path.join(data_root, '{}/context.txt'.format(stage))],
                "vocab_share_with": 0,
                "embedding_init_share_with": 0,
                "data_name": "context",
                "bos_token": '',
                "eos_token": '',
            },
            {  # target keywords
                "files": [os.path.join(data_root, '{}/keywords.txt'.format(stage))],
                "vocab_share_with": 0,
                "embedding_init_share_with": 0,
                "data_name": "keywords",
                "bos_token": '',
                "eos_token": '',
            },
            {  # label
                "files": [os.path.join(data_root, '{}/label.txt'.format(stage))],
                "data_type": "int",
                "data_name": "label"
            }
        ]
    }
    for stage in ['train','valid','test']
}


corpus_hparams = {
    "batch_size": _batch_size*2,
    "shuffle": False,
    "dataset":{
        "max_seq_length": _max_seq_len,
        "files": [os.path.join(data_root, 'corpus.txt')],
        "vocab_file": os.path.join(data_root, 'vocab.txt'),
        "data_name": "corpus"
    }
}


_keywords_path = 'tx_data/test/keywords_vocab.txt'
_keywords_candi = [x.strip() for x in open(_keywords_path, 'r').readlines()]
_keywords_num = len(_keywords_candi)
_keywords_dict = {}
for i in range(_keywords_num):
    _keywords_dict[_keywords_candi[i]] = i
