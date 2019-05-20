# -*- coding: utf-8 -*-
from gensim import corpora, similarities
from gensim.models import LsiModel


def main(JDK, url, title, query):

    dictionary = corpora.Dictionary.load('./LSI/data/lsi_dictionary.dict')
    index = similarities.Similarity.load('./LSI/data/lsi_index.index')
    lsi = LsiModel.load('./LSI/data/lsi.model')

    vec_bow = dictionary.doc2bow(query)
    vec_lsi = lsi[vec_bow]

    sims = index[vec_lsi]
    sort_sims = sorted(enumerate(sims), key=lambda item: -item[1])
    result = []
    for i in range(10):
        dic = {'url': url[sort_sims[i][0]].strip('\n'), 'JDK': JDK[sort_sims[i][0]].strip('\n'),
               'title': title[sort_sims[i][0]].strip('\n'), 'score': sort_sims[i][1]}
        result.append(dic)

    return result
