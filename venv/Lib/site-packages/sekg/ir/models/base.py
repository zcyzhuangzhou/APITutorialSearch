import abc
import os
import shutil
from pathlib import Path

from sekg.ir.doc.wrapper import PreprocessMultiFieldDocumentCollection


class DocRetrievalResult:
    """
    this class is wrapper for document retrieval result. Containing doc_id, doc_name,score.
    """

    def __init__(self, doc_id, ranking, score, doc_name, **extra_info):
        """
        create a wrapper for document search result.
        :param doc_id: the doc id.
        :param ranking: the ranking of the doc.
        :param score: the score of this document
        :param doc_name: the name of this document, could be None and duplicate
        :param extra_info: some extra info with this document
        """
        self.doc_id = doc_id
        self.doc_name = doc_name
        self.score = score
        self.ranking = ranking
        self.extra_info = extra_info

    def __repr__(self):
        return "<DocumentRetrieval result: rank=%d id=%d score=%r name=%s>" % (
            self.ranking, self.doc_id, self.score, self.doc_name,)


# todo: make the documentSimModel could train from normal document collection
class DocumentSimModel(metaclass=abc.ABCMeta):
    """
    the basic class for Model that computes the similarity score between query and documents.
    This is responsible for training the documentSim model and save and load the models, cache the model.
    And it could get the document similar to query by sim score.
    the sim score is between [0,1].
    Ps. each model is save under a directory. And all model file related to this model is put in this dir.
    """

    def __init__(self, name, model_dir_path, **config):
        """
        init the graph search model.
        :param name: the name of the model, just name the model instance, not other meaning.
        :param model_dir_path: the mode save dir path
        :param config: the config of the document similar model, eg. the weight of sub models
        """
        self.name = name
        self.model_dir_path = model_dir_path
        self.config = config
        self.preprocess_doc_collection = None
        # use to cache the score vector, score vector is a vector v
        # v=[0.5,2.0,3.0], v[0] means that the document 'd' ,
        # whose index is 0 in document collection, its score with query is 0.5.
        self.query_2_score_vector_cache = {}
        # sorted_index_scores is that score_vector sort by score from hign to low and keep the document index.
        self.query_2_sorted_index_scores_cache = {}
        self.doc_id_2_doc_index_map = dict()

    @abc.abstractmethod
    def init_model(self):
        """
        init the model, need to implement by each model
        :return:
        """
        print("start init the model")

    @abc.abstractmethod
    def init_model_as_submodel(self):
        """
        init the model as submodel.
        only load some some neccessary data for compute get_full_entity_score_vec
        :return:
        """
        print("start init model as sub_model ")

    def search(self, query, top_num=10, valid_doc_id_set=None):
        """
        search some document by query, return top score document by similar. and the document id must in the given id set.
        if valid_doc_id_set is not given. any document could be retrieved.
        :param valid_doc_id_set: the doc id set. the id of the returned document, its doc_id must in this id set.
        if None, any document could be returned.
        :param top_num: the number of return result, if the top_num =0, return all result in valid_doc_id_set
        :param query: a str, stand for the query
        :return: list of DocRetrievalResult.
        """
        if valid_doc_id_set is None:
            valid_doc_id_set = {}

        sorted_index_scores = self.get_sorted_index_doc_scores(query)
        retrieval_results = self.filter_retrieve_document(sorted_index_scores,
                                                          top_n=top_num,
                                                          valid_doc_id_set=valid_doc_id_set)

        return retrieval_results

    @abc.abstractmethod
    def get_full_doc_score_vec(self, query):
        """
        score vector is a vector v=[0.5,2.0,3.0], v[0] means that the document 'd',
         whose index is 0 in document collection, its score with query is 0.5.
        :param query: a str stands for the query.
        :return: get all document similar score with query as a numpy vector.
        """
        pass

    @abc.abstractmethod
    def train_from_doc_collection_with_preprocessor(self, doc_collection: PreprocessMultiFieldDocumentCollection,
                                                    **config):
        """
        train the doc collection with preprocess document collection
        :param doc_collection: doc_collection with preprocessor
        :return: a model instance
        """

        print("train_from_doc_collection_with_preprocessor config=%r" % config)
        self.preprocess_doc_collection = doc_collection
        self.config = config

    @classmethod
    def model_type(cls):
        """
        get the model type name
        :return: the model type name
        """
        return cls.__name__

    @classmethod
    def train(cls, model_dir_path, doc_collection: PreprocessMultiFieldDocumentCollection, **config):
        model = cls(cls.__name__, model_dir_path)
        model.train_from_doc_collection_with_preprocessor(doc_collection, **config)
        model.save(model_dir_path)
        return model

    @classmethod
    def load(cls, model_dir_path, **config):
        """
        load the model from a dir.
        :param model_dir_path: the dir saved a model
        :return: a model instance
        """
        model = cls.__create_from_a_exist_model_dir(config, model_dir_path)
        model.init_model()
        return model

    @classmethod
    def load_as_submodel(cls, model_dir_path, **config):
        """
        load the model from a dir.
        :param model_dir_path: the dir saved a model
        :return: a model instance
        """
        model = cls.__create_from_a_exist_model_dir(config, model_dir_path)
        model.init_model_as_submodel()
        return model

    @classmethod
    def __create_from_a_exist_model_dir(cls, config, model_dir_path):
        if model_dir_path is None:
            raise Exception("the model dir is None!!!")
        if not Path(model_dir_path).exists():
            raise Exception("the model dir %s is not exist" % model_dir_path)
        if not Path(model_dir_path).is_dir():
            raise Exception("the model dir %s is not not dir!!!" % model_dir_path)
        print("load the model from %s config=%r " % (model_dir_path, config))
        model = cls(cls.__name__, model_dir_path, **config)
        return model

    def save(self, model_dir_path):
        """
        save the model to a dir.
        :param model_dir_path: the dir to save a model
        :return:
        """
        if model_dir_path is None:
            raise Exception("the model dir is None!!!")

        if not Path(model_dir_path).exists():
            print("the model dir %s is not exist, creating" % model_dir_path)
            Path(model_dir_path).mkdir(parents=True, exist_ok=True)
        if not Path(model_dir_path).is_dir():
            raise Exception("the model dir %s is not not dir!!!" % model_dir_path)

        if len(os.listdir(model_dir_path)) != 0:
            print("the model dir %s is not empty before saving" % model_dir_path)
            shutil.rmtree(model_dir_path)
        print("save the model path")
        self.model_dir_path = model_dir_path

    def get_valid_doc_index_set(self, valid_doc_id_set):
        valid_doc_index_set = set()
        for doc_id in valid_doc_id_set:
            if doc_id in self.doc_id_2_doc_index_map:
                valid_doc_index_set.add(self.doc_id_2_doc_index_map[doc_id])
        return valid_doc_index_set

    def score_vec_index2doc_id(self, index):
        """
        parse the index to entity id
        :param index: the index of the entity object in similarity_vec int value
        :return: the entity id , integer, not the str
        """
        doc = self.score_vec_index2doc(index)
        if doc is None:
            return None
        return doc.get_document_id()

    def doc_id2doc(self, doc_id):
        """
        parse the entity_id to entity doc object (MultiFieldDocument)
        :return: the entity doc object(MultiFieldDocument)
        """
        return self.preprocess_doc_collection.get_raw_doc_by_id(doc_id)

    def score_vec_index2doc(self, index):
        """
        parse the index to entity id
        :param index: the index of the entity object in similarity_vec int value
        :return: MultiFieldDocument
        """
        return self.preprocess_doc_collection.get_raw_doc_by_index(index)

    def cache_entity_score_vector(self, query, index_scores):
        self.query_2_score_vector_cache[query] = index_scores

    def get_cache_score_vector(self, query):
        if query in self.query_2_score_vector_cache.keys():
            return self.query_2_score_vector_cache[query]
        return None

    def cache_sorted_index_scores(self, query, sorted_index_scores):
        self.query_2_sorted_index_scores_cache[query] = sorted_index_scores

    def get_cache_sorted_index_scores(self, query):
        if query in self.query_2_sorted_index_scores_cache.keys():
            return self.query_2_sorted_index_scores_cache[query]
        return None

    def get_doc_id_with_score(self, query):
        """
        get list of (doc_id,score) tuple.
        :param query:
        :return: a list
        """
        score_vector = self.get_full_doc_score_vec(query)
        doc_id_scores = [(self.score_vec_index2doc_id(index), score) for index, score in enumerate(score_vector)]
        return doc_id_scores

    def get_sorted_index_doc_scores(self, query):
        sorted_index_scores = self.get_cache_sorted_index_scores(query)
        if sorted_index_scores is None:
            score_vector = self.get_full_doc_score_vec(query)
            index_scores = [(index, score) for index, score in enumerate(score_vector)]
            sorted_index_scores = sorted(index_scores, key=lambda item: item[1], reverse=True)
            self.cache_sorted_index_scores(query, sorted_index_scores)

        return sorted_index_scores

    def filter_retrieve_document(self, sorted_index_scores, top_n, valid_doc_id_set=None):
        """
        filter the retrieve result.
        First, remove all doc not in valid doc id set(if valid doc id set is given, otherwise the all doc are keep).
        Second. get the specific num of document with top score
        :param sorted_index_scores: list of (index:int,score:double), it is sorted by the score from from high to low.
        :param top_n:if top n=0, the num of result is not limit.
        :param valid_doc_id_set: the valid doc id set.
        :return: a list of DocRetrievalResult, sorted by score
        """
        if not valid_doc_id_set:
            retrieval_results = self.__get_retrieval_result(sorted_index_scores, top_n)
        else:
            retrieval_results = self.__get_valid_retrieval_result(sorted_index_scores, top_n, valid_doc_id_set)

        return retrieval_results

    @staticmethod
    def __create_entity_retrieval_result(document_id, entity_document, ranking, score):
        single_result = DocRetrievalResult(doc_id=document_id, ranking=ranking, score=score,
                                           doc_name=entity_document.get_name(), entity_document=entity_document)
        return single_result

    def __get_valid_retrieval_result(self, sorted_index_scores, top_n, valid_doc_id_set):
        valid_doc_index_set = self.get_valid_doc_index_set(valid_doc_id_set)
        if top_n == 0:
            # return all valid result.
            retrieval_results = []
            ranking = 1
            for (doc_index, score) in sorted_index_scores:
                entity_document = self.score_vec_index2doc(doc_index)
                if entity_document is None:
                    continue
                if len(valid_doc_index_set) > 0 and doc_index not in valid_doc_index_set:
                    continue
                document_id = entity_document.get_document_id()
                if document_id not in valid_doc_id_set:
                    continue

                single_result = self.__create_entity_retrieval_result(document_id, entity_document, ranking, score)

                retrieval_results.append(single_result)
                ranking += 1

            return retrieval_results
        else:
            retrieval_results = []
            ranking = 1
            for (doc_index, score) in sorted_index_scores:
                entity_document = self.score_vec_index2doc(doc_index)
                if entity_document is None:
                    continue
                if len(valid_doc_index_set) > 0 and doc_index not in valid_doc_index_set:
                    continue
                document_id = entity_document.get_document_id()
                if document_id not in valid_doc_id_set:
                    continue
                single_result = self.__create_entity_retrieval_result(document_id, entity_document, ranking, score)

                retrieval_results.append(single_result)
                ranking += 1

                if ranking > top_n:
                    break
            return retrieval_results

    def __get_retrieval_result(self, sorted_index_scores, top_n):
        if top_n == 0:
            retrieval_results = []
            ranking = 1
            for (doc_index, score) in sorted_index_scores:
                entity_document = self.score_vec_index2doc(doc_index)
                if entity_document is None:
                    continue
                document_id = entity_document.get_document_id()

                single_result = self.__create_entity_retrieval_result(document_id, entity_document, ranking, score)
                retrieval_results.append(single_result)
                ranking += 1

            return retrieval_results
        else:
            retrieval_results = []
            ranking = 1
            for (doc_index, score) in sorted_index_scores[:top_n]:
                entity_document = self.score_vec_index2doc(doc_index)
                if entity_document is None:
                    continue
                document_id = entity_document.get_document_id()
                single_result = self.__create_entity_retrieval_result(document_id, entity_document, ranking, score)
                retrieval_results.append(single_result)
                ranking += 1
            return retrieval_results
