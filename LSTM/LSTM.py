# -*- coding: utf-8 -*-
# 基础包
import gensim
import pandas as pd
import keras
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import itertools
keras.backend.clear_session()
'''
本配置文件用于以服务形式调用预训练好的孪生网络预测新句对的相似度
'''


def test_data(embedding_dim, max_seq_length, embedding_dict, lstm_model, JDK, url, title, query):

    sen1 = query

    dataframe = pd.DataFrame({'question1': ["".join(sen1)]})

    for q in ['question1']:
        dataframe[q + '_n'] = dataframe[q]

    # 将测试集词向量化
    test_df = make_w2v_embeddings('en', embedding_dict, dataframe, embedding_dim=embedding_dim)

    # 预处理
    X_test = split_and_zero_padding(test_df, max_seq_length)

    sen2_arr = np.load('./LSTM/data/sen2_arr.npy')

    temp = np.array([X_test['left'][0]]*2084)

    prediction = lstm_model.predict([temp, sen2_arr])

    sort_sims = sorted(enumerate(prediction), key=lambda item: -item[1])

    result = []
    for i in range(10):
        dic = {'url': url[sort_sims[i][0]].strip('\n'), 'JDK': JDK[sort_sims[i][0]].strip('\n'),
               'title': title[sort_sims[i][0]].strip('\n'), 'score': sort_sims[i][1][0]}
        result.append(dic)
    return result


# -----------------主函数----------------- #
def main(JDK, url, title, query, lstm_model):

    # 加载LSTM模型
    embedding_path = './LSTM/data/Processed_JDK_SO.word2vec'
    embedding_dim = 300
    max_seq_length = 32

    embedding_dict = gensim.models.keyedvectors.Word2VecKeyedVectors.load(embedding_path)

    result = []
    result = test_data(embedding_dim, max_seq_length, embedding_dict, lstm_model, JDK, url, title, query)

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
