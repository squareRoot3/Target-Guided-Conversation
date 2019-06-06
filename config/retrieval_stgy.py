_hidden_size = 200
_code_len = 200
_save_path = 'save/retrieval/model_1'
_max_epoch = 10

source_encoder_hparams = {
    "encoder_minor_type": "UnidirectionalRNNEncoder",
    "encoder_minor_hparams": {
        "rnn_cell": {
            "type": "GRUCell",
            "kwargs": {
                "num_units": _hidden_size,
            },
        },
    },
    "encoder_major_type": "UnidirectionalRNNEncoder",
    "encoder_major_hparams": {
        "rnn_cell": {
            "type": "GRUCell",
            "kwargs": {
                "num_units": _hidden_size,
            },
        }
    }
}

target_encoder_hparams = {
    "rnn_cell": {
        "type": "GRUCell",
        "kwargs": {
            "num_units": _hidden_size,
        },
    }
}

opt_hparams = {
    "optimizer": {
        "type": "AdamOptimizer",
        "kwargs": {
            "learning_rate": 0.001,
        }
    },
}