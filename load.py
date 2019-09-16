# -*- coding: utf-8 -*-
import numpy as np
import pickle
import random
def atisfold():
    f = open('data/data_set.pkl')
    train_set, test_set, dicts = pickle.load(f)
    embedding = pickle.load(open('data/embedding.pkl'))
    return train_set, test_set,dicts,embedding

def pad_sentences(sentences, padding_word=0, forced_sequence_length=None):
    if forced_sequence_length is None:
        sequence_length=max(len(x) for x in sentences)
    else:
        sequence_length=forced_sequence_length
    padded_sentences=[]
    for i in range(len(sentences)):
        sentence=sentences[i]
        num_padding=sequence_length-len(sentence)
        if num_padding<0:
            padded_sentence=sentence[0:sequence_length]
        else:
            padded_sentence=sentence+[int(padding_word)]*num_padding

        padded_sentences.append(padded_sentence)

    return padded_sentences








