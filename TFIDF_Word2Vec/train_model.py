import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim import corpora, models
import gensim
from sekg.util.vector_util import VectorUtil


# 训练TF-IDF加权的Word2Vec模型
def train_tfidf_weight_w2v_model(corpus_clean_text, embedding_size, dictionary, tfidf, w2v_model):
    tfidf_w2v_model = gensim.models.keyedvectors.Word2VecKeyedVectors(vector_size=embedding_size)
    tfidf_vector_list = []
    tfidf_vector_index_str_list = []
    for index, doc_clean_text in enumerate(corpus_clean_text):
        vector = get_tfidf_w2v_vec(doc_clean_text, dictionary, tfidf, w2v_model)
        tfidf_vector_list.append(vector)
        tfidf_vector_index_str_list.append(str(index))
    tfidf_w2v_model.add(entities=tfidf_vector_index_str_list, weights=tfidf_vector_list, replace=True)
    tfidf_w2v_model.save('./data/tfidf-w2v.model')
    return tfidf_w2v_model


# 获得TF-IDF加权的词向量
def get_tfidf_w2v_vec(word_str_list, dict, tfidf_model, w2v_model):
    doc_word_vectors = []
    doc_bow = dict.doc2bow(word_str_list)
    doc_tfidf = tfidf_model[doc_bow]
    sorted_doc_tfidf = sorted(doc_tfidf, key=lambda x: x[1], reverse=True)
    sorted_doc_tfidf = sorted_doc_tfidf[:200]

    weight = []
    for word_id, tf_tfidf in sorted_doc_tfidf:
        word = dict[word_id]
        if word not in w2v_model.wv.vocab:
            continue
        doc_word_vectors.append(w2v_model.wv.__getitem__(word))
        weight.append(tf_tfidf)

    avg_vector = VectorUtil.get_weight_mean_vec(doc_word_vectors, weight)
    return avg_vector


# 训练TF-IDF模型
def trian_tfidf(texts):
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    tfidf = models.TfidfModel(corpus)
    tfidf.save('./data/tfidf.model')
    return dictionary, tfidf


# 训练Word2Vec模型
def train_word2vec():
    JDKdoc = open('./data/Processed_JDK_clean_text.txt', 'r', encoding='utf-8')
    word2vec = Word2Vec(LineSentence(JDKdoc), min_count=1, size=300)
    word2vec.save('./data/word2vec.model')
    return word2vec


if __name__ == "__main__":
    filePath = './data/Processed_JDK_clean_text.txt'
    courses = []
    with open(filePath, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            courses.append(line.split())

    dictionary, tfidf = trian_tfidf(courses)
    dictionary.save('./data/tfidf-w2v_dictionary.dict')
    word2vec = train_word2vec()
    tfidf_w2v_model = train_tfidf_weight_w2v_model(corpus_clean_text=courses, embedding_size=300, dictionary=dictionary,
                                                   tfidf=tfidf, w2v_model=word2vec)

    f.close()
