import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim import corpora, models, similarities


# 训练TF-IDF模型
def trian_tfidf(texts):
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    tfidf = models.TfidfModel(corpus)
    index = similarities.MatrixSimilarity(tfidf[corpus])

    tfidf.save('./data/tfidf.model')
    dictionary.save('./data/tfidf_dictionary.dict')
    index.save('./data/tfidf_index.index')


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

    trian_tfidf(courses)
    word2vec = train_word2vec()

    f.close()
