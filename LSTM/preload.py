# -*- coding: utf-8 -*-
# 基础包
import gensim
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import itertools
'''
本配置文件用于以服务形式调用预训练好的孪生网络预测新句对的相似度
'''


def test_data(max_seq_length, embedding_dict):
    X_test = []
    with open('E:/PycharmProjects/FlaskAPISearch/LSTM/data/Processed_JDK_clean_text.txt', 'r', encoding='utf-8') as df:
        for line2 in df.readlines():
            sen2 = line2
            dataframe = pd.DataFrame({'question2': ["".join(sen2)]})

            for q in ['question2']:
                dataframe[q + '_n'] = dataframe[q]

            # 将测试集词向量化
            test_df = make_w2v_embeddings(embedding_dict, dataframe)

            # 预处理
            X_test.append(split_and_zero_padding(test_df, max_seq_length))
    df.close()
    return X_test


def make_w2v_embeddings(word2vec, df):  # 将词转化为词向量
    vocabs = {}  # 词序号
    vocabs_cnt = 0  # 词个数计数器

    for index, row in df.iterrows():
        for question in ['question2']:
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
    X = {'right': df['question2_n']}

    # 调整到规定长度
    for dataset, side in itertools.product([X], ['right']):
        dataset[side] = pad_sequences(dataset[side], padding='pre', truncating='post', maxlen=max_seq_length)
        # print(dataset)

    return dataset


# -----------------主函数----------------- #
if __name__ == '__main__':
    # 加载LSTM模型
    embedding_path = 'E:/PycharmProjects/FlaskAPISearch/LSTM/data/Processed_JDK_SO.word2vec'

    max_seq_length = 32

    embedding_dict = gensim.models.keyedvectors.Word2VecKeyedVectors.load(embedding_path)

    X_test = test_data(max_seq_length, embedding_dict)
    sen2_temp = []
    for i in range(len(X_test)):
        sen2_temp.append(X_test[i]['right'][0])

    sen2_arr = np.array(sen2_temp)
    print(sen2_arr)
    np.save('./data/sen2_arr.npy', sen2_arr)
