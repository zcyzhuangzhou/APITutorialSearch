# -*- coding: utf-8 -*-
import gensim
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


def main(JDK, url, title, query):
    w2v = gensim.models.keyedvectors.Word2VecKeyedVectors.load('./Word2Vec/data/w2v_10.model')
    ave_w2v = gensim.models.keyedvectors.Word2VecKeyedVectors.load('./Word2Vec/data/ave_w2v_10.model')

    query_vec = get_avg_w2v_vec(query, w2v)
    print(query_vec)
    full_entity_score_vec = ave_w2v.similar_by_vector(query_vec, topn=False)
    print(len(full_entity_score_vec))
    sort_sims = sorted(enumerate(full_entity_score_vec), key=lambda item: -item[1])
    print(len(sort_sims))
    print(sort_sims)
    # print(len(sort_sims))
    result = []
    for i in range(10):
        dic = {'url': url[sort_sims[i][0]].strip('\n'), 'JDK': JDK[sort_sims[i][0]].strip('\n'),
               'title': title[sort_sims[i][0]].strip('\n'),  'score': sort_sims[i][1]}
        result.append(dic)

    return result
