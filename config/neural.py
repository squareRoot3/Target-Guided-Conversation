_hidden_size = 200
_code_len = 800
_save_path = 'save/neural/model_1'
_neural_save_path = 'save/neural/keyword_1'
_max_epoch = 10

neural_opt_hparams = {
    "optimizer": {
        "type": "AdamOptimizer",
        "kwargs": {
            "learning_rate": 0.005,
        }
    },
    "learning_rate_decay": {
        "type": "inverse_time_decay",
        "kwargs": {
            "decay_steps": 1600,
            "decay_rate": 0.8
        },
        "start_decay_step": 0,
        "end_decay_step": 16000,
    },
}

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

context_encoder_hparams = {
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
    }
}
