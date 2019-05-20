# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim.models import TfidfModel
from gensim import corpora, models
import gensim
from sekg.util.vector_util import VectorUtil


# 获得TF-IDF加权的词向量
def get_tfidf_w2v_vec(word_str_list, dict, tfidf_model, w2v_model):
    doc_word_vectors = []
    doc_bow = dict.doc2bow(word_str_list)
    doc_tfidf = tfidf_model[doc_bow]
    sorted_doc_tfidf = sorted(doc_tfidf, key=lambda x: x[1], reverse=True)
    sorted_doc_tfidf = sorted_doc_tfidf[:100]

    weight = []
    for word_id, tf_tfidf in sorted_doc_tfidf:
        word = dict[word_id]
        if word not in w2v_model.wv.vocab:
            continue
        doc_word_vectors.append(w2v_model.wv.__getitem__(word))
        weight.append(tf_tfidf)

    avg_vector = VectorUtil.get_weight_mean_vec(doc_word_vectors, weight)
    return avg_vector


def main(JDK, url, title, query):
    dictionary = corpora.Dictionary.load('./TFIDF_Word2Vec/data/tfidf-w2v_dictionary.dict')
    tfidf = TfidfModel.load('./TFIDF_Word2Vec/data/tfidf.model')
    word2vec = gensim.models.keyedvectors.Word2VecKeyedVectors.load('./TFIDF_Word2Vec/data/word2vec.model')
    tfidf_w2v_model = models.keyedvectors.Word2VecKeyedVectors.load('./TFIDF_Word2Vec/data/tfidf-w2v.model')

    query_vec = get_tfidf_w2v_vec(query, dictionary, tfidf, word2vec)
    full_entity_score_vec = tfidf_w2v_model.similar_by_vector(query_vec, topn=False)
    sort_sims = sorted(enumerate(full_entity_score_vec), key=lambda item: -item[1])

    result = []
    for i in range(10):
        dic = {'url': url[sort_sims[i][0]].strip('\n'), 'JDK': JDK[sort_sims[i][0]].strip('\n'),
               'title': title[sort_sims[i][0]].strip('\n'), 'score': sort_sims[i][1]}
        result.append(dic)

    return result
