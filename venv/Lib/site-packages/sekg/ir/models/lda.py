#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os

import gensim

from sekg.ir.doc.wrapper import PreprocessMultiFieldDocumentCollection
from sekg.ir.models.base import DocumentSimModel
from sekg.util.file import DirUtil


class LDAModel(DocumentSimModel):
    __lda_model_name__ = "lda.model"
    __corpus_name__ = "corpus.mm"
    __dictionary_name__ = "dictionary.dict"
    __lda_sim_index__ = "LDA.sim.index"

    __entity_collection_name__ = "doc.pre.collection"
    DEFAULT_TOPIC_NUM = 1000

    def __init__(self, name, model_dir_path, **config):

        """
        init the lsi model with
        :param model_dir_path:
        """
        super().__init__(name, model_dir_path, **config)
        self.lda_model = None
        self.corpus = None
        self.dict = None
        self.similarity_lda_index = None
        self.__init_sub_model_path__()
        self.num_topics = LDAModel.DEFAULT_TOPIC_NUM
        self.preprocessor = None

        self.preprocess_doc_collection = None

    def __init_sub_model_path__(self):
        if self.model_dir_path is None:
            return
        # init the paths of models
        self.lda_model_path = os.path.join(self.model_dir_path, self.__lda_model_name__)
        self.corpus_path = os.path.join(self.model_dir_path, self.__corpus_name__)
        self.dictionary_path = os.path.join(self.model_dir_path, self.__dictionary_name__)

        self.entity_collection_path = os.path.join(self.model_dir_path, self.__entity_collection_name__)

        self.sim_index_dir = os.path.join(self.model_dir_path, "index")

        self.sim_index_path = os.path.join(self.sim_index_dir, self.__lda_sim_index__)

        DirUtil.create_file_dir(self.model_dir_path)
        DirUtil.create_file_dir(self.sim_index_dir)

    def init_model(self):
        """
               init the model
               :return:
               """
        self.__init_sub_model_path__()
        print("loading the LDA models")
        self.lda_model = gensim.models.LdaModel.load(self.lda_model_path)
        self.dict = gensim.corpora.Dictionary.load(self.dictionary_path)
        print("All document number: ", self.dict.num_docs)
        print("All words number: ", self.dict.num_pos)
        print("load LSI models done")
        preprocess_doc_collection = PreprocessMultiFieldDocumentCollection.load(self.entity_collection_path)
        self.__init_document_collection(preprocess_doc_collection)
        self.__init_index()

    def init_model_as_submodel(self):
        """
        init the model
        :return:
        """
        self.__init_sub_model_path__()
        print("loading the LDA models")
        self.lda_model = gensim.models.LdaModel.load(self.lda_model_path)
        self.corpus = gensim.corpora.MmCorpus(self.corpus_path)
        self.dict = gensim.corpora.Dictionary.load(self.dictionary_path)
        print("All document number: ", self.dict.num_docs)
        print("All words number: ", self.dict.num_pos)
        print("load LSI models done")
        preprocess_doc_collection = PreprocessMultiFieldDocumentCollection.load(self.entity_collection_path)
        self.preprocessor = preprocess_doc_collection.get_preprocessor()
        self.field_set = self.preprocess_doc_collection.get_field_set()
        self.__init_index()

    def __init_document_collection(self, preprocess_doc_collection):
        self.preprocess_doc_collection = preprocess_doc_collection
        self.preprocessor = self.preprocess_doc_collection.get_preprocessor()
        self.field_set = self.preprocess_doc_collection.get_field_set()

    def train_from_document_collection(self, preprocess_document_collection, num_topics=400):
        print("start training")
        self.num_topics = num_topics
        self.__init_document_collection(preprocess_document_collection)
        corpus_clean_text = []
        preprocess_multi_field_doc_list = preprocess_document_collection.get_all_preprocess_document_list()

        for docno, multi_field_doc in enumerate(preprocess_multi_field_doc_list):
            corpus_clean_text.append(multi_field_doc.get_document_text_words())
            # print("process doc %d" % docno)

        print("corpus len=%d" % len(corpus_clean_text))

        print("Dictionary init...")

        self.dict = gensim.corpora.Dictionary(corpus_clean_text)
        print("Dictionary init complete ")

        self.corpus = [self.dict.doc2bow(text) for text in corpus_clean_text]

        print("LDA Training...")

        self.lda_model = gensim.models.LdaModel(corpus=self.corpus,
                                                id2word=self.dict,
                                                num_topics=num_topics)
        print("LDA Train complete")

    def train_from_doc_collection_with_preprocessor(self, doc_collection, num_topics=400, **config):
        # TODO: FIX THE PARAMETER, MAKE THE PARAMETER INPUT COULD BE MULTIPLE CHOICE
        if isinstance(doc_collection, PreprocessMultiFieldDocumentCollection):
            self.train_from_document_collection(preprocess_document_collection=doc_collection, num_topics=num_topics)

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

        self.lda_model.save(self.lda_model_path)
        print("LDA Training finish , save to %s" % self.lda_model_path)

        print("LDA similarity index init ...")
        self.similarity_lda_index = gensim.similarities.Similarity(self.sim_index_path,
                                                                   corpus=self.lda_model[self.corpus],
                                                                   num_features=self.num_topics)
        print("LDA similarity index complete")

        self.similarity_lda_index.save(self.sim_index_path)
        print("LDA similarity index finish , save to %s" % self.sim_index_path)

        print("entity collection saving...")
        self.preprocess_doc_collection.save(self.entity_collection_path)
        print(
            "entity collection finish saving , save to %s, %r" % (
                self.entity_collection_path, self.preprocess_doc_collection))

    def get_full_doc_score_vec(self, query):
        full_entity_score_vec = self.get_cache_score_vector(query)
        if full_entity_score_vec is not None:
            return full_entity_score_vec

        query_words = self.preprocessor.clean(query)
        query_bow, query_oov_dict = self.dict.doc2bow(query_words, return_missing=True)
        query_lda_vec = self.lda_model[query_bow]
        full_entity_score_vec = self.similarity_lda_index[query_lda_vec]
        self.cache_entity_score_vector(query, full_entity_score_vec)

        return full_entity_score_vec

    def __init_index(self):
        try:
            self.similarity_lda_index = gensim.similarities.Similarity.load(self.sim_index_path)
            self.get_full_doc_score_vec("test")

        except FileNotFoundError:
            print("the sim index is useless, generate again")
            self.similarity_lda_index = gensim.similarities.Similarity(self.sim_index_path,
                                                                       corpus=self.lda_model[self.corpus],
                                                                       num_features=self.num_topics)
            self.similarity_lda_index.save(self.sim_index_path)
