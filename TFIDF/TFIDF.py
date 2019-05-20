# -*- coding: utf-8 -*-
from gensim import corpora, similarities
from gensim.models import TfidfModel


def main(JDK, url, title, query):

    dictionary = corpora.Dictionary.load('./TFIDF/data/tfidf_dictionary.dict')
    index = similarities.Similarity.load('./TFIDF/data/tfidf_index.index')
    tfidf = TfidfModel.load('./TFIDF/data/tfidf.model')

    vec_bow = dictionary.doc2bow(query)
    vec_tfidf = tfidf[vec_bow]

    sims = index[vec_tfidf]
    sort_sims = sorted(enumerate(sims), key=lambda item: -item[1])
    result = []
    for i in range(10):
        dic = {'url': url[sort_sims[i][0]].strip('\n'), 'JDK': JDK[sort_sims[i][0]].strip('\n'),
               'title': title[sort_sims[i][0]].strip('\n'), 'score': sort_sims[i][1]}
        result.append(dic)

    return result
