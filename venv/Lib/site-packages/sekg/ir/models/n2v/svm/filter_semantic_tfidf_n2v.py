#!/usr/bin/python
# -*- coding: UTF-8 -*-
import pickle
from pathlib import Path

import numpy as np
from gensim.corpora import Dictionary, MmCorpus
from gensim.models import TfidfModel
from gensim.models.keyedvectors import Word2VecKeyedVectors

from sekg.graph.util.name_searcher import KGNameSearcher
from sekg.ir.doc.wrapper import PreprocessMultiFieldDocumentCollection
from sekg.ir.models.base import DocumentSimModel
from sekg.ir.models.util.load import LoadUtil
from sekg.util.vector_util import VectorUtil


class FilterSemanticTFIDFNode2VectorModel(DocumentSimModel):
    """
    filter some node by a pretrained Search Model(TFIDFSearchModel)"""
    __corpus_name__ = 'corpus.mm'
    __n2v_model_name__ = "node2vec.model"
    __dictionary_name__ = 'dictionary.dict'
    __entity_collection_name__ = 'entity.collection'
    __kg_name_searcher_name__ = 'kg.namesearcher'
    __graph_data_name__ = "kg.graph"
    __tfidf_model_name__ = "tfidf.model"

    __model_config__ = "model.config"
    __doc_sim_model_dir_name__ = "doc_sim_model"

    DEFAULT_EMBEDDING_SIZE = 100
    MIN_NODE_SIM_SCORE = 0.5
    MAX_CANDIDATE_NODE_NUMBER_FOR_ONE_WORD = 10000

    def __init__(self, name, model_dir_path, **config):

        """
        init the lsi model with
        :param model_dir_path:
        """
        super().__init__(name, model_dir_path, **config)
        self.node2vec_model = None

        self.corpus = None
        self.dict = None
        self.tfidf_model = None
        self.similarity_index = None

        self.kg_name_searcher = None
        self.graph_data = None

        self.__init_sub_model_path__()

        self.preprocessor = None
        self.preprocess_doc_collection = None
        self.NP_VECTOR_NOT_EXIST = None
        self.embedding_size = 0
        self.__init_embedding_size(self.DEFAULT_EMBEDDING_SIZE)

        self.doc_sim_model_path = None
        self.doc_sim_model_class = None
        self.doc_sim_model = None

    def load_extra_config(self):
        """
        load some config from pickle file
        :return:
        """
        with open(self.extra_config_path, 'rb') as aq:
            extra_config = pickle.loads(aq.read())

        self.doc_sim_model_class = extra_config["doc_sim_model_class"]
        self.embedding_size = extra_config["embedding_size"]

    def save_extra_config(self):
        """
        save some config into pickle file
        :return:
        """
        extra_config = {"doc_sim_model_class": self.doc_sim_model_class, "embedding_size": self.embedding_size}

        with open(self.extra_config_path, 'wb') as out:
            out.write(pickle.dumps(extra_config))

    def __init_sub_model_path__(self):
        if self.model_dir_path is None:
            return

        model_dir_path_obj = Path(self.model_dir_path)

        # init the paths of models
        self.corpus_path = str(model_dir_path_obj / self.__corpus_name__)

        self.dictionary_path = str(model_dir_path_obj / self.__dictionary_name__)
        self.n2v_model_path = str(model_dir_path_obj / self.__n2v_model_name__)
        self.graph_data_path = str(model_dir_path_obj / self.__graph_data_name__)
        self.entity_collection_path = str(model_dir_path_obj / self.__entity_collection_name__)
        self.kg_name_searcher_path = str(model_dir_path_obj / self.__kg_name_searcher_name__)

        self.tfidf_model_path = str(model_dir_path_obj / self.__tfidf_model_name__)
        self.doc_sim_model_path = str(model_dir_path_obj / self.__doc_sim_model_dir_name__)

        Path(self.doc_sim_model_path).mkdir(exist_ok=True, parents=True)

        self.extra_config_path = str(model_dir_path_obj / self.__model_config__)

    def __init_document_collection(self, preprocess_doc_collection):
        self.preprocess_doc_collection = preprocess_doc_collection
        self.preprocessor = preprocess_doc_collection.get_preprocessor()
        self.field_set = preprocess_doc_collection.get_field_set()

    def __init_embedding_size(self, embedding_size):
        self.embedding_size = embedding_size
        self.NP_VECTOR_NOT_EXIST = np.zeros(embedding_size)
        self.NP_VECTOR_NOT_EXIST[0] = 1e-07

    def init_model_as_submodel(self):
        """
        init the model
        :return:
        """
        print("loading doc_collection")
        self.__init_sub_model_path__()
        preprocess_doc_collection = PreprocessMultiFieldDocumentCollection.load(self.entity_collection_path)
        self.preprocessor = preprocess_doc_collection.get_preprocessor()

        print("loading the tfidf model models")
        self.tfidf_model = TfidfModel.load(self.tfidf_model_path)

        print("loading the Node2vec models")
        self.node2vec_model = Word2VecKeyedVectors.load(self.n2v_model_path)

        print("loading the KGNameSearcher models")

        self.kg_name_searcher = KGNameSearcher.load(self.kg_name_searcher_path)

        self.dict = Dictionary.load(self.dictionary_path)
        print("All document number: ", self.dict.num_docs)
        print("All words number: ", self.dict.num_pos)

        self.load_extra_config()
        self.doc_sim_model = self.doc_sim_model_class.load_as_submodel(self.doc_sim_model_path)

        print("loading %r model from %s" % (self.doc_sim_model_class.model_type(), self.doc_sim_model_path))

    def init_model(self):
        """
        init the model
        :return:
        """
        print("loading doc_collection")
        self.__init_sub_model_path__()
        preprocess_doc_collection = PreprocessMultiFieldDocumentCollection.load(self.entity_collection_path)
        self.__init_document_collection(preprocess_doc_collection)

        print("loading the tfidf model models")
        self.tfidf_model = TfidfModel.load(self.tfidf_model_path)

        print("loading the Node2vec models")
        self.node2vec_model = Word2VecKeyedVectors.load(self.n2v_model_path)

        print("loading the KGNameSearcher models")

        self.kg_name_searcher = KGNameSearcher.load(self.kg_name_searcher_path)

        # todo: load the index
        self.corpus = MmCorpus(self.corpus_path)
        self.dict = Dictionary.load(self.dictionary_path)
        print("All document number: ", self.dict.num_docs)
        print("All words number: ", self.dict.num_pos)

        self.load_extra_config()
        self.doc_sim_model = self.doc_sim_model_class.load_as_submodel(self.doc_sim_model_path)

        print("loading %r model from %s" % (self.doc_sim_model_class.model_type(), self.doc_sim_model_path))

    def get_full_doc_score_vec(self, query):
        full_entity_score_vec = self.get_cache_score_vector(query)
        if full_entity_score_vec is not None:
            return full_entity_score_vec

        query_vec = self.compute_query_graph_vec(query)

        full_entity_score_vec = self.node2vec_model.similar_by_vector(query_vec, topn=False)
        full_entity_score_vec = (full_entity_score_vec + 1) / 2
        self.cache_entity_score_vector(query, full_entity_score_vec)
        return full_entity_score_vec

    def compute_query_graph_vec(self, query):
        words = self.preprocessor.extract_words_for_query(query)
        word_tfidf_weight_dict = {}

        query_bow, query_oov_dict = self.dict.doc2bow(words, return_missing=True)
        query_tfidf_vec = self.tfidf_model[query_bow]

        for word_id, tfidf_value in query_tfidf_vec:
            word = self.dict[word_id]
            word_tfidf_weight_dict[word] = tfidf_value

        tfidf_full_entity_score_vec = self.doc_sim_model.get_full_doc_score_vec(query)
        index_scores = [(index, score) for index, score in enumerate(tfidf_full_entity_score_vec) if
                        score > self.MIN_NODE_SIM_SCORE]

        valid_id_scores_list = [(self.score_vec_index2doc_id(index), score) for (index, score) in index_scores]
        valid_id_scores_map = {}
        for entity_id, score in valid_id_scores_list:
            valid_id_scores_map[entity_id] = score

        valid_id_set = set([entity_id for (entity_id, score) in valid_id_scores_list])

        entity_weight_dict = {}
        for word in word_tfidf_weight_dict.keys():
            entity_id_set = self.kg_name_searcher.search_by_name(word)
            entity_id_set = valid_id_set.intersection(entity_id_set)

            if len(entity_id_set) == 0:
                continue
            if len(entity_id_set) > self.MAX_CANDIDATE_NODE_NUMBER_FOR_ONE_WORD:
                print("keyword search for %s is %d, too many" % (word, len(entity_id_set)))
                continue

            temp_weight = word_tfidf_weight_dict[word] / len(entity_id_set)

            for entity_id in entity_id_set:
                if entity_id not in entity_weight_dict.keys():
                    entity_weight_dict[entity_id] = 0.0
                entity_weight_dict[entity_id] = entity_weight_dict[entity_id] + temp_weight

        for entity_id in entity_weight_dict.keys():
            entity_weight_dict[entity_id] = entity_weight_dict[entity_id] * valid_id_scores_map[entity_id]

        query_vec = self.get_weight_graph_vec(entity_weight_dict)
        return query_vec

    def train_from_doc_collection_with_preprocessor(self, doc_collection: PreprocessMultiFieldDocumentCollection,
                                                    embedding_size=DEFAULT_EMBEDDING_SIZE,
                                                    pretrain_node2vec_path=None,
                                                    graph_data_path=None,
                                                    graph_data=None,
                                                    tfidf_model_path=None,
                                                    tfidf_model=None,
                                                    kg_name_searcher_path=None,
                                                    kg_name_searcher=None,
                                                    doc_sim_model_path=None,
                                                    doc_sim_model_class=None):

        print("start training")
        self.__init_embedding_size(embedding_size)
        self.__init_document_collection(doc_collection)

        corpus_clean_text = []
        preprocess_multi_field_doc_list = doc_collection.get_all_preprocess_document_list()
        for docno, multi_field_doc in enumerate(preprocess_multi_field_doc_list):
            corpus_clean_text.append(multi_field_doc.get_document_text_words())

        print("corpus len=%d" % len(corpus_clean_text))

        print("Dictionary init...")
        self.dict = Dictionary(corpus_clean_text)
        print("Dictionary init complete ")

        print("parse to bow corpus")
        self.corpus = [self.dict.doc2bow(text) for text in corpus_clean_text]
        print("parse to bow corpus complete")

        self.node2vec_model = LoadUtil.load_node2vec_for_doc_collection(doc_collection,
                                                                        pretrain_node2vec_path=pretrain_node2vec_path,
                                                                        embedding_size=embedding_size
                                                                        )

        self.kg_name_searcher = LoadUtil.load_kg_name_searcher(kg_name_searcher=kg_name_searcher,
                                                               kg_name_searcher_path=kg_name_searcher_path)
        self.graph_data = LoadUtil.load_graph_data(graph_data_path=graph_data_path, graph_data=graph_data)

        self.tfidf_model = LoadUtil.load_tfidf_model(tfidf_model=tfidf_model, corpus=self.corpus,
                                                     tfidf_model_path=tfidf_model_path, dict=self.dict)
        self.doc_sim_model_class = doc_sim_model_class
        self.doc_sim_model = doc_sim_model_class.load(doc_sim_model_path)

    def get_weight_graph_vec(self, entity_weight_dict):
        """
        get the average word2vec for list of str
        :param entity_weight_dict:  list of str
        :return: np.array()
        """
        if len(entity_weight_dict.keys()) == 0:
            return self.NP_VECTOR_NOT_EXIST

        valid_entity_id_str = []
        entity_weight_list = []

        for entity_id, w in entity_weight_dict.items():
            entity_id_str = str(entity_id)
            if entity_id_str not in self.node2vec_model.vocab:
                continue
            valid_entity_id_str.append(entity_id_str)
            entity_weight_list.append(w)

        if len(valid_entity_id_str) == 0:
            return self.NP_VECTOR_NOT_EXIST

        entity_vector_list = self.node2vec_model[valid_entity_id_str]
        if entity_vector_list is []:
            return self.NP_VECTOR_NOT_EXIST
        avg_vector = VectorUtil.get_weight_mean_vec(vector_list=entity_vector_list, weight_list=entity_weight_list)

        return avg_vector

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

        MmCorpus.serialize(self.corpus_path, self.corpus)
        print("save the corpus to %s" % self.corpus_path)

        print("entity collection saving...")
        self.preprocess_doc_collection.save(self.entity_collection_path)
        print(
            "entity collection finish saving , save to %s, %r" % (
                self.entity_collection_path, self.preprocess_doc_collection))

        print("kg_name_searcher saving...")

        self.kg_name_searcher.save(self.kg_name_searcher_path)
        print("kg_name_searcher saving done...")

        print("node2vec_model saving...")
        self.node2vec_model.save(self.n2v_model_path)
        print("node2vec Training finish , save to %s" % self.n2v_model_path)

        print("graph data saving...")
        self.graph_data.save(self.graph_data_path)
        print("graph data saving done...")

        print("TFIDF model saving...")
        self.tfidf_model.save(self.tfidf_model_path)
        print("TFIDF saving finish , save to %s" % self.tfidf_model_path)

        self.doc_sim_model.save(self.doc_sim_model_path)
        self.save_extra_config()
