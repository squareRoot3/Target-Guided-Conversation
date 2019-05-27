# Target-Guided Open-Domain Conversation

### Requirement
nltk = 3.4  
tensoflow = 1.12  
You also need to install [Texar](https://github.com/asyml/texar).


### Usage

#### Data Preparation
You need to download data from [google drive](https://drive.google.com/file/d/1oTjOQjm7iiUitOPLCmlkXOCbEPoSWDPX/view?usp=sharing)
and unzip it into `preprocess/convai2`. Then run the following command:
```shell
python preprocess/prepare_data.py
```
By default, the processed data will be put in the `tx_data` directory.

#### Turn-level Supervised Learning
In this project there are 5 different types of agents, including the kernel/neural/matrix/retrieval/retrieval_stgy agent,
 which are all discribed in our paper. You can modify the configration of each agent in the `config` directory.

To train the kernel/neural/matrix agent, you need to first train/test the keyword prediction module, 
and then train/test the retrieval module of each agent specified by the `--agent` parameter.

```shell
python train.py --mode train_kw --agent kernel
python train.py --mode train --agent kernel
python train.py --mode test --agent kernel
```

The retrieval agent and the retrieval_stgy agent share the same retrival module. You are only need to train one of them:

```shell
python train.py --mode train --agent retrieval
python train.py --mode test --agent retrieval
```

#### Target-guided Conversation

After turn-level training, you can start target-guided conversation (human evaluation) with 
the kernel/neural/matrix/retrieval/retrieval_stgy  agent specified by the `--agent` parameter.

```shell
python chat.py --agent kernel
```

You can also watch the simulation of the target-guided conversation 
between the retrieval agent pretending the user and the kernel/neural/matrix/retrieval_stgy agent specified by the `--agent` parameter.

```shell
python simulate.py --agent kernel
```
