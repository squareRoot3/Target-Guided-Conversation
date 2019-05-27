# Target-Guided Open-Domain Conversation

This is the code for the following paper:

[Target-Guided Open-Domain Conversation](.)  
*Jianheng Tang, Tiancheng Zhao, Chengyan Xiong, Xiaodan Liang, Eric Xing, Zhiting Hu; ACL 2019*

### Requirement

- `nltk==3.4`  
- `tensoflow==1.12`   
- `texar>=0.2.1` ([Texar](https://github.com/asyml/texar))

### Usage

#### Data Preparation
The dataset developed in the paper is on [google drive](https://drive.google.com/file/d/1oTjOQjm7iiUitOPLCmlkXOCbEPoSWDPX/view?usp=sharing). Download 
and unzip it into `preprocess/convai2`. Then run the following command:
```shell
cd preprocess
python prepare_data.py
```
By default, the processed data will be put in the `tx_data` directory.

#### Turn-level Supervised Learning
In this project there are 5 different types of agents, including the kernel/neural/matrix/retrieval/retrieval_stgy agent,
 which are all discribed in the paper. You can modify the configration of each agent in the `config` directory.

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
