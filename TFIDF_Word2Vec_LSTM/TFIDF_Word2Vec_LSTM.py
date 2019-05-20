# -*- coding: utf-8 -*-
import gensim
from gensim import similarities, corpora
from gensim.models import TfidfModel
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import itertools


def get_avg_w2v_vec(word_str_list, w2v_model):
    valid_word_str_list = []

    for word in word_str_list:
        if word not in w2v_model.wv.vocab:
            continue
        valid_word_str_list.append(word)

    doc_word_vectors = w2v_model[valid_word_str_list]

    avg_vector = np.mean(doc_word_vectors, axis=0)
    return avg_vector


def main(JDK, url, title, query, lstm_model):

    # 加载word2vec模型
    w2v = gensim.models.keyedvectors.Word2VecKeyedVectors.load('./TFIDF_Word2Vec_LSTM/data/w2v.model')
    ave_w2v = gensim.models.keyedvectors.Word2VecKeyedVectors.load('./TFIDF_Word2Vec_LSTM/data/ave_w2v.model')

    # 加载LSTM模型
    embedding_path = './TFIDF_Word2Vec_LSTM/data/Processed_JDK_SO.word2vec'
    embedding_dim = 300
    max_seq_length = 32
    embedding_dict = gensim.models.keyedvectors.Word2VecKeyedVectors.load(embedding_path)

    # 加载tfidf模型
    dictionary = corpora.Dictionary.load('./TFIDF_Word2Vec_LSTM/data/tfidf_dictionary.dict')
    index = similarities.Similarity.load('./TFIDF_Word2Vec_LSTM/data/tfidf_index.index')
    tfidf = TfidfModel.load('./TFIDF_Word2Vec_LSTM/data/tfidf.model')

    # tfidf得分计算
    vec_bow = dictionary.doc2bow(query)
    vec_tfidf = tfidf[vec_bow]
    tfidf_sims = index[vec_tfidf]

    # word2vec得分计算
    query_vec = get_avg_w2v_vec(query, w2v)
    w2v_sims = ave_w2v.similar_by_vector(query_vec, topn=False)

    # LSTM得分计算
    sen1 = query

    dataframe = pd.DataFrame({'question1': ["".join(sen1)]})

    for q in ['question1']:
        dataframe[q + '_n'] = dataframe[q]

    # 将测试集词向量化
    test_df = make_w2v_embeddings('en', embedding_dict, dataframe, embedding_dim=embedding_dim)

    # 预处理
    X_test = split_and_zero_padding(test_df, max_seq_length)

    sen2_arr = np.load('E:/PycharmProjects/FlaskAPISearch/TFIDF_Word2Vec_LSTM/data/sen2_arr.npy')

    lstm_sims = lstm_model.predict([np.array([X_test['left'][0]] * 2084), sen2_arr], batch_size=2084)

    # tfidf + word2vec + lstm得分
    tfidf_w2v_lstm_sims = 0.43 * np.array(tfidf_sims[:]) + 0.17 * np.array(w2v_sims[:]) + 0.4 * np.array(lstm_sims[:][0])
    sort_tfidf_w2v_lstm_sims = sorted(enumerate(tfidf_w2v_lstm_sims), key=lambda item: -item[1])

    result = []
    for i in range(10):
        dic = {'url': url[sort_tfidf_w2v_lstm_sims[i][0]].strip('\n'), 'JDK': JDK[sort_tfidf_w2v_lstm_sims[i][0]].strip('\n'),
               'title': title[sort_tfidf_w2v_lstm_sims[i][0]].strip('\n'), 'score': sort_tfidf_w2v_lstm_sims[i][1]}
        result.append(dic)

    return result


def make_w2v_embeddings(flag, word2vec, df, embedding_dim):  # 将词转化为词向量
    vocabs = {}  # 词序号
    vocabs_cnt = 0  # 词个数计数器

    for index, row in df.iterrows():
        for question in ['question1']:
            q2n = []  # q2n -> question to numbers representation

            for word in row[question]:
                if word not in word2vec.wv.vocab:  # OOV的词放入不能用词向量表示的字典中，value为1
                    continue
                if word not in vocabs:  # 非OOV词，提取出对应的id
                    vocabs_cnt += 1
                    vocabs[word] = vocabs_cnt
                    q2n.append(vocabs_cnt)
                else:
                    q2n.append(vocabs[word])
            df.at[index, question + '_n'] = q2n

    return df


def split_and_zero_padding(df, max_seq_length):  # 调整tokens长度

    # 训练集矩阵转换成字典
    X = {'left': df['question1_n']}

    # 调整到规定长度
    for dataset, side in itertools.product([X], ['left']):
        dataset[side] = pad_sequences(dataset[side], padding='pre', truncating='post', maxlen=max_seq_length)
        # print(type(dataset['left']), dataset)

    return dataset
