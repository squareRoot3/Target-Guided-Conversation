# Target-Guided Open-Domain Conversation

This is the code for the following paper:

[Target-Guided Open-Domain Conversation](http://arxiv.org/abs/1905.11553)  
*Jianheng Tang, Tiancheng Zhao, Chenyan Xiong, Xiaodan Liang, Eric Xing, Zhiting Hu; ACL 2019*

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

The retrieval agent and the retrieval_stgy agent share the same retrival module. You only need to train one of them:

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
A demo of target-guided conversation (user is required to input the response after `HUMAN:`)

```
--------Session 0 --------
START:  hello, how are you doing
HUMAN:  hi i am great just finishing up some homework. how about you
AGENT:  just got done eating some chicken and rice after i got done with my gym session
HUMAN:  i am gonna eat some vegetable for dinner
AGENT:  so you are vegetarian ? i am from los angeles so thereare lots of vegetarian restaurants
HUMAN:  i am not a vegetarian . fried chicken is my favorite.
AGENT:  oh boy , when i go to mcdonald i always get fries for my 2 dogs
Successfully chat to the target 'mcdonald'.
```

You can also watch the simulation of the target-guided conversation 
between the retrieval agent pretending the user and the kernel/neural/matrix/retrieval_stgy agent specified by the `--agent` parameter. Simulation times are defined by the `--times` parameter. The program will calculate the success times and average turns in the end.

```shell
python simulate.py --agent kernel --times 100
```

A demo of simulation:
```
--------Session 0 --------
START: hey hey good morning to you
retrieval_agent:  fine thanks . what do you do for a living ?
kernel_agent:  rewarding job , i work at a hospital
Keyword: job, Similarity: 0.58
Successfully chat to the target 'hospital'.

...
--------Session 99 --------
START: hey hows it going ? i'm just cooking a steak
retrieval_agent:  i'm thinking of a bbq sandwich for lunch
kernel_agent:  nice i love to cook but now its just me and the fur babies
Keyword: baby, Similarity: 0.45
retrieval_agent:  i love bagels however i own a dry cleaners
kernel_agent:  i love animals felix my cat and my dog emmy
Keyword: cat, Similarity: 0.56
retrieval_agent:  sounds awesome i have all kind of pets my family own a farm
kernel_agent:  i love blue as well even my hair is blue
Keyword: blue, Similarity: 1.00
Successfully chat to the target 'blue'.

success time 83, average turns 4.28
```
