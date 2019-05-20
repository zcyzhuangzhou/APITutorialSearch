import numpy as np
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.models.keyedvectors import Word2VecKeyedVectors

from sekg.graph.exporter.graph_data import GraphData
from sekg.graph.util.name_searcher import KGNameSearcher
from sekg.ir.doc.wrapper import PreprocessMultiFieldDocumentCollection


class LoadUtil:
    DEFAULT_EMBEDDING_SIZE = 100

    @staticmethod
    def load_graph_data(graph_data_path, graph_data: GraphData):
        if graph_data is None:
            if graph_data_path is not None:
                print("graph_data is given, loading")

                return GraphData.load(graph_data_path)
            else:
                print("graph_data is not given, training")
                # todo: implement training the GraphData here, currently is only load a pretrain KGNameSearcher  model
        else:
            return graph_data

    @staticmethod
    def load_kg_name_searcher(kg_name_searcher: KGNameSearcher, kg_name_searcher_path):
        if kg_name_searcher:
            return kg_name_searcher

        if kg_name_searcher is None:

            if kg_name_searcher_path is not None:
                print("pretrain_search_model_path is given, loading")

                return KGNameSearcher.load(kg_name_searcher_path)
            else:
                print("pretrain_search_model_path is not given, training, but the training is not implemented yet")
                # todo: implement training the KGNameSearcher here, currently is only load a pretrain KGNameSearcher  model

    @staticmethod
    def load_tfidf_model(tfidf_model_path, tfidf_model: TfidfModel, dict: Dictionary, corpus):
        if tfidf_model:
            return tfidf_model

        if tfidf_model_path is not None:
            print("pretrain tfidf path is given, loading")
            return TfidfModel.load(tfidf_model_path)

        else:
            print("tfidf Training...")
            tfidf_model = TfidfModel(corpus=corpus, id2word=dict)
            print("tfidf Train complete")
            return tfidf_model

    @staticmethod
    def get_unknown_vector(embedding_size):
        unknwon_vector = np.zeros(embedding_size)
        unknwon_vector[0] = 1e-07
        return unknwon_vector

    @classmethod
    def load_node2vec_for_doc_collection(cls, preprocess_document_collection: PreprocessMultiFieldDocumentCollection,
                                         pretrain_node2vec_path=None,
                                         embedding_size=DEFAULT_EMBEDDING_SIZE,

                                         ):
        """
        train the node2vec model
        :param preprocess_document_collection:
        :param pretrain_node2vec_path:
        :param embedding_size:
        :return:
        """

        not_exist_vector = LoadUtil.get_unknown_vector(embedding_size)

        if pretrain_node2vec_path is not None:
            print("pretrained node2vec path path is given, loading")
            full_node2vec_model = Word2VecKeyedVectors.load(pretrain_node2vec_path)
            new_node2vec_model = Word2VecKeyedVectors(vector_size=embedding_size)

            doc_id_str_list = []
            vector_list = []

            invalid_doc_id_count = 0
            doc_list = preprocess_document_collection.get_all_preprocess_document_list()
            for doc in doc_list:
                doc_id_str = str(doc.get_document_id())
                if doc_id_str not in full_node2vec_model.vocab:
                    vector = not_exist_vector
                    invalid_doc_id_count = invalid_doc_id_count + 1
                else:
                    vector = full_node2vec_model[doc_id_str]

                doc_id_str_list.append(doc_id_str)
                vector_list.append(vector)

            if len(doc_id_str_list) != len(set(doc_id_str_list)):
                raise Exception("error when filter the node2vec, duplicate doc id in input doc collection!")

            new_node2vec_model.add(entities=doc_id_str_list, weights=vector_list, replace=True)
            print("full node2vec size=%d, invald id=%d new_size=%d" % (
                len(full_node2vec_model.vocab), invalid_doc_id_count, len(new_node2vec_model.vocab)))

            return new_node2vec_model

        else:
            print("pretrained node2vec path is not given, training")

            # print("Word2Vec Training...")
            # # using the CBOW model of word2vec, because we don't use to predict
            # w2v_model = gensim.models.Word2Vec(sentences=corpus_clean_text, size=embedding_size, min_count=1)
            # print("Word2Vec Train complete")
            # self.node2vec_model = w2v_model.wv
            # todo: implement training the node2vec here, currently is only load a pretrain node2ve  model
