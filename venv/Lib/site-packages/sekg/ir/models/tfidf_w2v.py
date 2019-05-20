#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os

import gensim
import numpy as np

from sekg.ir.doc.wrapper import PreprocessMultiFieldDocumentCollection
from sekg.ir.models.base import DocumentSimModel
from sekg.util.file import DirUtil
from sekg.util.vector_util import VectorUtil


class TFIDFW2VFLModel(DocumentSimModel):
    __tfidf_model_name__ = "tfidf.model"
    __corpus_name__ = 'corpus.mm'
    __w2v_model_name__ = "word2vec.model"
    __tfidf_w2v_model_name__ = "word2vec.tfidf.model"
    __dictionary_name__ = 'dictionary.dict'
    __sim_index__ = "tfidfw2v.sim.index"

    __entity_collection_name__ = "doc.pre.collection"
    DEFAULT_EMBEDDING_SIZE = 100
    DEFAULT_TFIDF_KEYWORD_NUM = 20

    def __init__(self, name, model_dir_path, **config):

        """
        init the lsi model with
        :param model_dir_path:
        """
        super().__init__(name, model_dir_path, **config)

        self.tfidf_model = None
        self.tfidf_model_path = None
        self.w2v_model = None
        self.tfidf_w2v_model = None
        self.corpus = None
        self.dict = None
        self.similarity_index = None
        self.__init_sub_model_path__()
        self.preprocessor = None
        self.preprocess_doc_collection = None

        self.NP_VECTOR_NOT_EXIST = None
        self.__init_embedding_size(TFIDFW2VFLModel.DEFAULT_EMBEDDING_SIZE)
        self.entity_collection_path = None

    def __init_sub_model_path__(self):
        if self.model_dir_path is None:
            return
            # init the paths of models
        self.corpus_path = os.path.join(self.model_dir_path, self.__corpus_name__)
        self.dictionary_path = os.path.join(self.model_dir_path, self.__dictionary_name__)
        if self.tfidf_model_path is None:
            self.tfidf_model_path = os.path.join(self.model_dir_path, self.__tfidf_model_name__)

        self.w2v_model_path = os.path.join(self.model_dir_path, self.__w2v_model_name__)
        self.tfidf_w2v_model_path = os.path.join(self.model_dir_path, self.__tfidf_w2v_model_name__)

        self.entity_collection_path = os.path.join(self.model_dir_path, self.__entity_collection_name__)

        self.sim_index_dir = os.path.join(self.model_dir_path, "index")

        self.sim_index_path = os.path.join(self.sim_index_dir, self.__sim_index__)

        DirUtil.create_file_dir(self.model_dir_path)
        DirUtil.create_file_dir(self.sim_index_dir)

    def __init_document_collection(self, preprocess_doc_collection):
        self.preprocess_doc_collection = preprocess_doc_collection
        self.preprocessor = preprocess_doc_collection.get_preprocessor()
        self.field_set = preprocess_doc_collection.get_field_set()

    def init_model_as_submodel(self):
        """
        init the model
        :return:
        """
        print("init_sub_model_path")
        self.__init_sub_model_path__()

        print("loading doc_collection")
        preprocess_doc_collection = PreprocessMultiFieldDocumentCollection.load(self.entity_collection_path)
        self.preprocessor = preprocess_doc_collection.get_preprocessor()
        self.field_set = preprocess_doc_collection.get_field_set()

        print("loading the tfidf model models")
        self.tfidf_model = gensim.models.TfidfModel.load(self.tfidf_model_path)
        print("loading the Word2vec models")
        self.w2v_model = gensim.models.keyedvectors.Word2VecKeyedVectors.load(self.w2v_model_path)
        print("loading the tfidf w2v models")
        self.tfidf_w2v_model = gensim.models.keyedvectors.Word2VecKeyedVectors.load(self.tfidf_w2v_model_path)

        self.dict = gensim.corpora.Dictionary.load(self.dictionary_path)
        print("All document number: ", self.dict.num_docs)
        print("All words number: ", self.dict.num_pos)
        print("load models done")

    def init_model(self):
        """
        init the model
        :return:
        """
        print("init_sub_model_path")
        self.__init_sub_model_path__()

        print("loading doc_collection")
        preprocess_doc_collection = PreprocessMultiFieldDocumentCollection.load(self.entity_collection_path)
        self.__init_document_collection(preprocess_doc_collection)

        print("loading the tfidf model models")
        self.tfidf_model = gensim.models.TfidfModel.load(self.tfidf_model_path)
        print("loading the Word2vec models")
        self.w2v_model = gensim.models.keyedvectors.Word2VecKeyedVectors.load(self.w2v_model_path)
        print("loading the tfidf w2v models")
        self.tfidf_w2v_model = gensim.models.keyedvectors.Word2VecKeyedVectors.load(self.tfidf_w2v_model_path)

        self.corpus = gensim.corpora.MmCorpus(self.corpus_path)
        self.dict = gensim.corpora.Dictionary.load(self.dictionary_path)
        print("All document number: ", self.dict.num_docs)
        print("All words number: ", self.dict.num_pos)
        print("load models done")

    def __init_embedding_size(self, embedding_size):
        self.embedding_size = embedding_size
        self.NP_VECTOR_NOT_EXIST = np.zeros(embedding_size)
        self.NP_VECTOR_NOT_EXIST[0] = 1e-07

    def get_full_doc_score_vec(self, query):
        full_entity_score_vec = self.get_cache_score_vector(query)
        if full_entity_score_vec is not None:
            return full_entity_score_vec

        query_words = self.preprocessor.clean(query)
        query_vec = self.get_tfidf_w2v_vec(query_words)
        full_entity_score_vec = self.tfidf_w2v_model.similar_by_vector(query_vec, topn=False)
        full_entity_score_vec = (full_entity_score_vec + 1) / 2
        # save the result to cache
        self.cache_entity_score_vector(query, full_entity_score_vec)
        return full_entity_score_vec

    def train_from_document_collection(self, preprocess_document_collection, embedding_size=100,
                                       pretrain_w2v_path=None, pretrain_tfidf_path=None, pretrain_dictionary_path=None):
        print("start training")
        self.__init_embedding_size(embedding_size)
        self.__init_document_collection(preprocess_document_collection)

        corpus_clean_text = []
        preprocess_multi_field_doc_list = preprocess_document_collection.get_all_preprocess_document_list()
        for docno, multi_field_doc in enumerate(preprocess_multi_field_doc_list):
            corpus_clean_text.append(multi_field_doc.get_document_text_words())

        print("corpus len=%d" % len(corpus_clean_text))

        if pretrain_dictionary_path is not None:
            print("pretrain pretrain dictionary path is given, loading")
            self.dict = gensim.corpora.Dictionary.load(pretrain_dictionary_path)
        else:
            print("Dictionary init...")
            self.dict = gensim.corpora.Dictionary(corpus_clean_text)
            print("Dictionary init complete")

        print("parse to bow corpus")
        self.corpus = [self.dict.doc2bow(text) for text in corpus_clean_text]
        print("parse to bow corpus complete")

        if pretrain_w2v_path is not None:
            print("pretrain Word2vec path is given, loading")
            self.w2v_model = gensim.models.keyedvectors.Word2VecKeyedVectors.load(pretrain_w2v_path)
        else:
            print("pretrain Word2vec path is not given, loading")

            print("Word2Vec Training...")
            w2v_model = gensim.models.Word2Vec(sentences=corpus_clean_text, size=embedding_size, min_count=1)
            print("Word2Vec Train complete")
            self.w2v_model = w2v_model.wv

        if pretrain_tfidf_path is not None:
            print("pretrain tfidf path is given, loading")
            self.tfidf_model = gensim.models.TfidfModel.load(pretrain_tfidf_path)

        else:
            print("tfidf Training...")
            self.tfidf_model = gensim.models.TfidfModel(corpus=self.corpus, id2word=self.dict)
            print("tfidf Train complete")

        self.tfidf_w2v_model = self.train_tfidf_weight_w2v_model(corpus_clean_text, embedding_size)

    def train_tfidf_weight_w2v_model(self, corpus_clean_text, embedding_size):
        self.__init_embedding_size(embedding_size=embedding_size)

        tfidf_w2v_model = gensim.models.keyedvectors.Word2VecKeyedVectors(vector_size=embedding_size)
        tfidf_vector_list = []
        tfidf_vector_index_str_list = []
        for index, doc_clean_text in enumerate(corpus_clean_text):
            vector = self.get_tfidf_w2v_vec(doc_clean_text)
            tfidf_vector_list.append(vector)
            tfidf_vector_index_str_list.append(str(index))
        tfidf_w2v_model.add(entities=tfidf_vector_index_str_list, weights=tfidf_vector_list, replace=True)

        return tfidf_w2v_model

    def get_tfidf_w2v_vec(self, word_str_list):
        """
        get the average word2vec for list of str
        :param word_str_list:  list of str
        :return: np.array()
        """
        doc_word_vectors = []
        doc_bow = self.dict.doc2bow(word_str_list)
        doc_tfidf = self.tfidf_model[doc_bow]
        sorted_doc_tfidf = sorted(doc_tfidf, key=lambda x: x[1], reverse=True)
        sorted_doc_tfidf = sorted_doc_tfidf[:self.DEFAULT_TFIDF_KEYWORD_NUM]

        weight = []
        for word_id, tf_tfidf in sorted_doc_tfidf:
            word = self.dict[word_id]
            if word not in self.w2v_model.vocab:
                continue
            doc_word_vectors.append(self.w2v_model[word])
            weight.append(tf_tfidf)
        if len(doc_word_vectors) == 0:
            return self.NP_VECTOR_NOT_EXIST
        if len(weight) == 0:
            return self.NP_VECTOR_NOT_EXIST
        try:
            avg_vector = VectorUtil.get_weight_mean_vec(doc_word_vectors, weight)
            return avg_vector
        except Exception as e:
            print(e)
            # print(doc_word_vectors)
            # print(weight)
            return self.NP_VECTOR_NOT_EXIST

    def train_from_doc_collection_with_preprocessor(self, doc_collection, **config):
        embedding_size = self.DEFAULT_EMBEDDING_SIZE
        pretrain_w2v_path = None
        pretrain_tfidf_path = None
        pretrain_dictionary_path = None

        if "embedding_size" in config.keys():
            embedding_size = config["embedding_size"]
        if "pretrain_w2v_path" in config.keys():
            pretrain_w2v_path = config["pretrain_w2v_path"]
        if "pretrain_tfidf_path" in config.keys():
            pretrain_tfidf_path = config["pretrain_tfidf_path"]
        if "pretrain_dictionary_path" in config.keys():
            pretrain_dictionary_path = config["pretrain_dictionary_path"]

        if isinstance(doc_collection, PreprocessMultiFieldDocumentCollection):
            self.train_from_document_collection(preprocess_document_collection=doc_collection,
                                                embedding_size=embedding_size,
                                                pretrain_w2v_path=pretrain_w2v_path,
                                                pretrain_tfidf_path=pretrain_tfidf_path,
                                                pretrain_dictionary_path=pretrain_dictionary_path
                                                )

    def save(self, model_dir_path):
        """
        save model to the model_dir_path
        :param model_dir_path: the dir to save the model
        :return:
        """
        super().save(model_dir_path)

        self.__init_sub_model_path__()

        self.dict.save(self.dictionary_path)

        print("build dictionary in %s" % self.dictionary_path)

        gensim.corpora.MmCorpus.serialize(self.corpus_path, self.corpus)
        print("save the corpus to %s" % self.corpus_path)

        self.w2v_model.save(self.w2v_model_path)
        print("Word2vec save finish , save to %s" % self.w2v_model_path)

        self.tfidf_model.save(self.tfidf_model_path)
        print("tfidf save finish , save to %s" % self.tfidf_model_path)

        self.tfidf_w2v_model.save(self.tfidf_w2v_model_path)
        print("TF IDF Word2vec Training finish , save to %s" % self.tfidf_w2v_model_path)

        print("entity collection saving...")
        self.preprocess_doc_collection.save(self.entity_collection_path)
        print(
            "entity collection finish saving , save to %s, %r" % (
                self.entity_collection_path, self.preprocess_doc_collection))
