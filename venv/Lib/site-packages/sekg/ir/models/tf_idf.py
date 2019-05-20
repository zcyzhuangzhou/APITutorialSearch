#!/usr/bin/python
# -*- coding: UTF-8 -*-
from pathlib import Path

import gensim

from sekg.ir.doc.wrapper import PreprocessMultiFieldDocumentCollection
from sekg.ir.models.base import DocumentSimModel


# todo: the index file will has problem when the model are copy to new path
class TFIDFModel(DocumentSimModel):
    """
    This class is used the TF-IDF model to compute the document similarity score.
    The implementation from TFIDF is from gensim.
    """
    __tfidf_model_name__ = "tfidf.model"
    __corpus_name__ = "corpus.mm"
    __dictionary_name__ = "corpus.dict"
    # the index file is created for speeding up the similarity score computing
    __tfidf_sim_index__ = "TFIDF.sim.index"
    # the name for the PreprocessMultiFieldDocumentCollection. store
    __doc_collection_name__ = "doc.pre.collection"

    def __init__(self, name, model_dir_path, **config):

        """
        init the lsi model with
        :param model_dir_path:
        """
        super().__init__(name, model_dir_path, **config)

        self.tfidf_model = None
        self.corpus = None
        self.dict = None
        self.similarity_tfidf_index = None
        self.__init_sub_model_path__()

        self.preprocessor = None
        self.preprocess_doc_collection = None

    def __init_sub_model_path__(self):
        if self.model_dir_path is None:
            return
        # init the paths of models
        model_dir = Path(self.model_dir_path)
        self.tfidf_model_path = str(model_dir / self.__tfidf_model_name__)
        self.corpus_path = str(model_dir / self.__corpus_name__)
        self.dictionary_path = str(model_dir / self.__dictionary_name__)
        self.entity_collection_path = str(model_dir / self.__doc_collection_name__)

        index_dir = model_dir / "index"
        self.sim_index_dir = str(index_dir)
        self.sim_index_path = str(index_dir / self.__tfidf_sim_index__)

        model_dir.mkdir(parents=True, exist_ok=True)
        index_dir.mkdir(parents=True, exist_ok=True)

    def init_model_as_submodel(self):
        """
        init the model
        :return:
        """
        self.__init_sub_model_path__()
        print("loading the TFIDF models")
        self.tfidf_model = gensim.models.TfidfModel.load(self.tfidf_model_path)
        self.dict = gensim.corpora.Dictionary.load(self.dictionary_path)
        self.similarity_tfidf_index = gensim.similarities.Similarity.load(self.sim_index_path)
        print("All document number: ", self.dict.num_docs)
        print("All words number: ", self.dict.num_pos)
        print("load TFIDF models done")
        preprocess_doc_collection = PreprocessMultiFieldDocumentCollection.load(self.entity_collection_path)
        self.preprocessor = preprocess_doc_collection.get_preprocessor()
        self.field_set = preprocess_doc_collection.get_field_set()
        self.__init_index()

    def init_model(self):
        """
        init the model
        :return:
        """
        self.__init_sub_model_path__()
        print("loading the TFIDF models")
        self.tfidf_model = gensim.models.TfidfModel.load(self.tfidf_model_path)
        self.corpus = gensim.corpora.MmCorpus(self.corpus_path)
        self.dict = gensim.corpora.Dictionary.load(self.dictionary_path)
        print("All document number: ", self.dict.num_docs)
        print("All words number: ", self.dict.num_pos)
        print("load TFIDF models done")
        preprocess_doc_collection = PreprocessMultiFieldDocumentCollection.load(self.entity_collection_path)
        self.__init_document_collection(preprocess_doc_collection)
        self.__init_index()

    def __init_document_collection(self, preprocess_doc_collection):
        self.preprocess_doc_collection = preprocess_doc_collection
        self.preprocessor = self.preprocess_doc_collection.get_preprocessor()
        self.field_set = self.preprocess_doc_collection.get_field_set()

    def train_from_document_collection(self, preprocess_document_collection, num_topics=1000,
                                       pretrain_tfidf_path=None,
                                       pretrain_dictionary_path=None):
        print("start training")
        self.num_topics = num_topics
        self.__init_document_collection(preprocess_document_collection)
        corpus_clean_text = []
        preprocess_multi_field_doc_list = preprocess_document_collection.get_all_preprocess_document_list()

        for docno, multi_field_doc in enumerate(preprocess_multi_field_doc_list):
            corpus_clean_text.append(multi_field_doc.get_document_text_words())
            # print("process doc %d" % docno)

        print("corpus len=%d" % len(corpus_clean_text))

        if pretrain_dictionary_path is not None:
            print("pretrain pretrain dictionary path is given, loading")
            self.dict = gensim.corpora.Dictionary.load(pretrain_dictionary_path)
        else:
            print("Dictionary init...")
            self.dict = gensim.corpora.Dictionary(corpus_clean_text)
            print("Dictionary init complete")

        self.corpus = [self.dict.doc2bow(text) for text in corpus_clean_text]

        if pretrain_tfidf_path is not None:
            print("pretrain tfidf path is given, loading")
            self.tfidf_model = gensim.models.TfidfModel.load(pretrain_tfidf_path)

        else:
            print("tfidf Training...")
            self.tfidf_model = gensim.models.TfidfModel(corpus=self.corpus, id2word=self.dict)
            print("tfidf Train complete")

    def train_from_doc_collection_with_preprocessor(self, doc_collection: PreprocessMultiFieldDocumentCollection,
                                                    **config):
        pretrain_tfidf_path = None
        pretrain_dictionary_path = None
        if "pretrain_tfidf_path" in config.keys():
            pretrain_tfidf_path = config["pretrain_tfidf_path"]
        if "pretrain_dictionary_path" in config.keys():
            pretrain_dictionary_path = config["pretrain_dictionary_path"]

        if isinstance(doc_collection, PreprocessMultiFieldDocumentCollection):
            self.train_from_document_collection(preprocess_document_collection=doc_collection,
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

        self.tfidf_model.save(self.tfidf_model_path)
        print("TFIDF Training finish , save to %s" % self.tfidf_model_path)

        print("TFIDF similarity index init ...")
        self.similarity_tfidf_index = gensim.similarities.Similarity(self.sim_index_path,
                                                                     corpus=self.tfidf_model[self.corpus],
                                                                     num_features=len(self.dict))
        print("TFIDF similarity index complete")

        self.similarity_tfidf_index.save(self.sim_index_path)
        print("TFIDF similarity index finish , save to %s" % self.sim_index_path)

        print("entity collection saving...")
        self.preprocess_doc_collection.save(self.entity_collection_path)
        print(
            "entity collection finish saving , save to %s, %r" % (
                self.entity_collection_path, self.preprocess_doc_collection))

    def get_full_doc_score_vec(self, query):
        """
        score vector is a vector v=[0.5,2.0,3.0], v[0] means that the document 'd' whose index is 0 in document collection, its score with query is 0.5.
        :param query: a str stands for the query.
        :return: get all document similar score with query as a numpy vector.
        """

        full_entity_score_vec = self.get_cache_score_vector(query)
        if full_entity_score_vec is not None:
            return full_entity_score_vec

        query_words = self.preprocessor.clean(query)
        query_bow, query_oov_dict = self.dict.doc2bow(query_words, return_missing=True)
        query_tfidf_vec = self.tfidf_model[query_bow]
        full_entity_score_vec = self.similarity_tfidf_index[query_tfidf_vec]

        self.cache_entity_score_vector(query, full_entity_score_vec)
        return full_entity_score_vec

    def __init_index(self):
        try:
            self.similarity_tfidf_index = gensim.similarities.Similarity.load(self.sim_index_path)
            self.get_full_doc_score_vec("add")
            # todo: change to some keyword must in the dict
        except FileNotFoundError:
            print("the sim index is useless, generate again")
            self.similarity_tfidf_index = gensim.similarities.Similarity(self.sim_index_path,
                                                                         corpus=self.tfidf_model[self.corpus],
                                                                         num_features=len(self.dict))

            self.similarity_tfidf_index.save(self.sim_index_path)
