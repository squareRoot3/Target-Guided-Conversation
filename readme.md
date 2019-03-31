# Target-guided Conversation

### Requirement
nltk = 3.4  
tensoflow = 1.9  
texar = 0.1  

### Data Preparation

You can download the processed data from [our Onedrive]() and train it directly. The data root path is defined in `data_config.py`.  

You can also use the source conversation corpus from [The Conversational Intelligence Challenge 2](http://convai.io/) and process it by the scripts in `preprocess`. We provide a copy of the source corpus in [our Onedrive]().

- Use processed data
  - [download website]()
  -  move processed data in 
- Use source data
  - [download website]()
  - move the data to `preprocess\convai2`
  -  run `data_maker.py`

### Usage

```shell
# train the keyword prediction module
python train.py -type kernel -mode train_kw

# train the retrieval module
python train.py -type kernel -mode train

# start target-guided open-domain conversation
python chat.py
```

### Todo
- add self-play (with the retrieval-baseline)
- adapt a larger corpus
- improve the topic guiding strategy of our agent

