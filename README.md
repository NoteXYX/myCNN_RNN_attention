# Keyphrase Extraction
Source codes of A Keyphrase Extraction method based on Multi-size Convolution Windows

## Preparation
You need to prepare  the pre-trained word vectors.
* Pre-trained word vectors. Download [GoogleNews-vectors-negative300.bin.gz](https://code.google.com/archive/p/word2vec/)


## Details
Multi-size CNN + Joint RNN model + attention

* data文件夹存储数据集

* checkpoints文件夹存储模型训练得到的参数

* main.py是主程序

* models/model.py定义了我们的模型

* models/bi_lstm_model.py 用双向lstm代替rnn

* load.py用于加载数据集

* tools.py定义了一些工具函数

## Requirement
tensorflow1.14.0 + nltk

