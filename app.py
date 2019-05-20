# -*- coding: utf-8 -*-
import string
import time
from keras import backend as K
from keras.layers import Layer
import keras
import nltk
from flask import Flask, render_template, request, redirect
import sys
from nltk.corpus import stopwords
sys.path.append('./LSI')
sys.path.append('./LDA')
sys.path.append('./TFIDF')
sys.path.append('./Word2Vec')
sys.path.append('./TFIDF_Word2Vec')
sys.path.append('./LSTM')
sys.path.append('./TFIDF_Word2Vec_LSTM')
import LSI
import LDA
import TFIDF
import Word2Vec
import TFIDF_Word2Vec
import LSTM
import TFIDF_Word2Vec_LSTM
keras.backend.clear_session()

app = Flask(__name__)


class ManDist(Layer):  # 封装成keras层的曼哈顿距离计算

    # 初始化ManDist层，此时不需要任何参数输入
    def __init__(self, **kwargs):
        self.result = None
        super(ManDist, self).__init__(**kwargs)

    # 自动建立ManDist层
    def build(self, input_shape):
        super(ManDist, self).build(input_shape)

    # 计算曼哈顿距离
    def call(self, x, **kwargs):
        self.result = K.exp(-K.sum(K.abs(x[0] - x[1]), axis=1, keepdims=True))
        return self.result

    # 返回结果
    def compute_output_shape(self, input_shape):
        return K.int_shape(self.result)


# 预加载文档
JDK = []
url = []
title = []
with open('./data/JDK_clean_text.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        JDK.append(line)
    f.close()
with open('./data/JDK_url.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        url.append(line)
    f.close()
with open('./data/JDK_title.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        title.append(line)
    f.close()

savepath = './LSTM/data/en_SiameseLSTM.h5'
lstm_model = keras.models.load_model(savepath, custom_objects={"ManDist": ManDist})
lstm_model._make_predict_function()


@app.route('/')
def root():
    return redirect('home')


@app.route('/home')
def home():
    return render_template('home.html')


@app.route('/model')
def model():
    return render_template('model.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/result', methods=['POST'])
def result():
    cpu_start = time.clock()
    query = request.values['query']
    print('query:', query)
    searchMethod = request.values['search']
    print('searchMethod:', searchMethod)

    # query处理
    query_token = list(nltk.word_tokenize(query.lower().strip()))  # 分词&小写
    english_stopwords = stopwords.words('english')
    query_stop = [word for word in query_token if not word in english_stopwords]  # 去停用词
    query_punct = [word for word in query_stop if not word in string.punctuation]  # 去标点
    st = nltk.LancasterStemmer()
    processed_query = [st.stem(word) for word in query_punct]  # 词干化

    if searchMethod == 'TFIDF':
        search_result = TFIDF.main(JDK, url, title, processed_query)
        search_result[0].update(searchMethod=searchMethod)
        search_result[0].update(query=query)
    elif searchMethod == 'LDA':
        search_result = LDA.main(JDK, url, title, processed_query)
        search_result[0].update(searchMethod=searchMethod)
        search_result[0].update(query=query)
    elif searchMethod == 'LSI':
        search_result = LSI.main(JDK, url, title, processed_query)
        search_result[0].update(searchMethod=searchMethod)
        search_result[0].update(query=query)
    elif searchMethod == 'Word2Vec':
        search_result = Word2Vec.main(JDK, url, title, processed_query)
        search_result[0].update(searchMethod=searchMethod)
        search_result[0].update(query=query)
    elif searchMethod == 'LSTM':
        search_result = LSTM.main(JDK, url, title, processed_query, lstm_model)
        search_result[0].update(searchMethod=searchMethod)
        search_result[0].update(query=query)
    elif searchMethod == 'TFIDF_Word2Vec':
        search_result = TFIDF_Word2Vec.main(JDK, url, title, processed_query)
        search_result[0].update(searchMethod=searchMethod)
        search_result[0].update(query=query)
    elif searchMethod == 'TFIDF_Word2Vec_LSTM':
        search_result = TFIDF_Word2Vec_LSTM.main(JDK, url, title, processed_query, lstm_model)
        search_result[0].update(searchMethod=searchMethod)
        search_result[0].update(query=query)

    cpu_end = time.clock()
    search_result[0].update(time=round(cpu_end - cpu_start, 2))
    return render_template('result.html', result=search_result)


if __name__ == '__main__':
    app.run(processes=7)
