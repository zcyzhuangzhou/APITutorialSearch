import logging
from gensim import corpora, models, similarities


# 训练LSI模型
def trian_lsi(texts):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    # 训练LSI模型
    lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=1000)
    index = similarities.MatrixSimilarity(lsi[corpus])

    lsi.save('./data/lsi.model')
    dictionary.save('./data/lsi_dictionary.dict')
    index.save('./data/lsi_index.index')


if __name__ == "__main__":
    filePath = './data/Processed_JDK_clean_text.txt'
    courses = []
    with open(filePath, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            courses.append(line.split())

    trian_lsi(courses)

    f.close()
