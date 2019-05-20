# -*- coding: utf-8 -*-
import gensim

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import logging
import numpy as np


def get_avg_w2v_vec(word_str_list, w2v_model):
    valid_word_str_list = []

    for word in word_str_list:
        if word not in w2v_model.wv.vocab:
            continue
        valid_word_str_list.append(word)

    doc_word_vectors = w2v_model[valid_word_str_list]

    avg_vector = np.mean(doc_word_vectors, axis=0)
    return avg_vector


def Train_Word2Vec():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    JDKdoc = open('./data/Processed_JDK_clean_text.txt', 'r', encoding='utf-8')

    model = Word2Vec(LineSentence(JDKdoc), min_count=1, size=300)
    model.save('./data/w2v.model')
    filePath = './data/Processed_JDK_clean_text.txt'
    courses = []
    with open(filePath, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            courses.append(line.split())

    w2v_model = gensim.models.keyedvectors.Word2VecKeyedVectors(vector_size=300)
    vector_list = []
    vector_index_str_list = []

    for index, doc_clean_text in enumerate(courses):
        vector = get_avg_w2v_vec(doc_clean_text, model)
        vector_list.append(vector)
        vector_index_str_list.append(str(index))
    w2v_model.add(entities=vector_index_str_list, weights=vector_list, replace=True)
    w2v_model.save('./data/ave_w2v.model')


if __name__ == '__main__':
    Train_Word2Vec()
