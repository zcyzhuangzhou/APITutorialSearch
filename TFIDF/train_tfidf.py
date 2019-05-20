import logging
from gensim import corpora, models, similarities


# 训练TF-IDF模型
def trian_tfidf(texts):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    tfidf = models.TfidfModel(corpus)
    index = similarities.MatrixSimilarity(tfidf[corpus])

    tfidf.save('./data/tfidf.model')
    dictionary.save('./data/tfidf_dictionary.dict')
    index.save('./data/tfidf_index.index')


if __name__ == "__main__":
    filePath = './data/Processed_JDK_clean_text.txt'
    courses = []
    with open(filePath, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            courses.append(line.split())

    trian_tfidf(courses)

    f.close()
