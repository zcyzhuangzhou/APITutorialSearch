# -*- coding: utf-8 -*-
from gensim import corpora, similarities
from gensim.models import LdaModel


def main(JDK, url, title, query):

    # 加载模型
    dictionary = corpora.Dictionary.load('./LDA/data/lda_dictionary.dict')
    index = similarities.Similarity.load('./LDA/data/lda_index.index')
    lda = LdaModel.load('./LDA/data/lda.model')

    vec_bow = dictionary.doc2bow(query)
    vec_lda = lda[vec_bow]

    sims = index[vec_lda]
    sort_sims = sorted(enumerate(sims), key=lambda item: -item[1])
    result = []
    for i in range(10):
        dic = {'url': url[sort_sims[i][0]].strip('\n'), 'JDK': JDK[sort_sims[i][0]].strip('\n'),
               'title': title[sort_sims[i][0]].strip('\n'), 'score': sort_sims[i][1]}
        result.append(dic)

    return result
