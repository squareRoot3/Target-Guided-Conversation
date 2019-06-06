_hidden_size = 200
_code_len = 800
_save_path = 'save/matrix/model_1'
_matrix_save_path = 'save/matrix/matrix_1.pk'
_max_epoch = 10

_vocab_path = 'tx_data/vocab.txt'
_vocab = [x.strip() for x in open(_vocab_path, 'r').readlines()]
_vocab_size = len(_vocab)

source_encoder_hparams = {
    "encoder_minor_type": "BidirectionalRNNEncoder",
    "encoder_minor_hparams": {
        "rnn_cell_fw": {
            "type": "GRUCell",
            "kwargs": {
                "num_units": _hidden_size,
            },
        },
        "rnn_cell_share_config": True
    },
    "encoder_major_type": "UnidirectionalRNNEncoder",
    "encoder_major_hparams": {
        "rnn_cell": {
            "type": "GRUCell",
            "kwargs": {
                "num_units": _hidden_size*2,
            },
        }
    }
}

target_encoder_hparams = {
    "rnn_cell_fw": {
        "type": "GRUCell",
        "kwargs": {
            "num_units": _hidden_size,
        },
    },
    "rnn_cell_share_config": True
}

target_kwencoder_hparams = {
    "rnn_cell_fw": {
        "type": "GRUCell",
        "kwargs": {
            "num_units": _hidden_size,
        },
    },
    "rnn_cell_share_config": True
}

opt_hparams = {
    "optimizer": {
        "type": "AdamOptimizer",
        "kwargs": {
            "learning_rate": 0.001,
        }
    }
}