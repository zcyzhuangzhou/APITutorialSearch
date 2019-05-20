import logging
from gensim import corpora, models, similarities


# 训练LDA模型
def trian_lda(texts):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    # 训练LDA模型
    lda = models.LdaModel(corpus, id2word=dictionary, num_topics=1000)
    index = similarities.MatrixSimilarity(lda[corpus])

    dictionary.save('./data/lda_dictionary.dict')
    lda.save('./data/lda.model')
    index.save('./data/lda_index.index')


if __name__ == "__main__":
    filePath = './data/Processed_JDK_clean_text.txt'
    courses = []
    with open(filePath, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            courses.append(line.split())

    trian_lda(courses)

    f.close()
