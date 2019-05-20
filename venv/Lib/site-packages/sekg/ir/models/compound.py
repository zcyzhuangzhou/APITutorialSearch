#!/usr/bin/python
# -*- coding: UTF-8 -*-
import pickle
from pathlib import Path

from sekg.ir.doc.wrapper import PreprocessMultiFieldDocumentCollection
from sekg.ir.models.base import DocumentSimModel
from sekg.util.vector_util import VectorUtil


# todo: change the model in here to save the model to local path
class CompoundSearchModel(DocumentSimModel):
    __sub_search_model_config__ = "submodel.config"
    __entity_collection_name__ = 'entity.collection'

    def __init__(self, name, model_dir_path, **config):

        """
        init the lsi model with
        :param model_dir_path:
        """
        super().__init__(name, model_dir_path, **config)
        self.sub_search_model_config = []
        self.preprocess_doc_collection = None
        self.model_list = []
        self.model_weight_list = []

        self.__init_sub_model_path__()

    def save(self, model_dir_path):
        super().save(model_dir_path)
        self.__init_sub_model_path__()
        with open(self.sub_search_model_config_path, 'wb') as out:
            out.write(pickle.dumps(self.sub_search_model_config))
        print("saving the sub_search_model_config complete")

        print("entity collection saving...")
        self.preprocess_doc_collection.save(self.entity_collection_path)
        print(
            "entity collection finish saving , save to %s, %r" % (
                self.entity_collection_path, self.preprocess_doc_collection))

    def __init_sub_model_path__(self):
        if self.model_dir_path is None:
            return
        Path(self.model_dir_path).mkdir(exist_ok=True, parents=True)
        # init the paths of models
        self.sub_search_model_config_path = str(Path(self.model_dir_path) / self.__sub_search_model_config__)
        self.entity_collection_path = str(Path(self.model_dir_path) / self.__entity_collection_name__)

    def init_model_as_submodel(self):
        self.init_model()

    def init_model(self):
        self.__init_sub_model_path__()

        with open(self.sub_search_model_config_path, 'rb') as aq:
            self.sub_search_model_config = pickle.loads(aq.read())
        self.init_all_sub_model()
        if self.preprocess_doc_collection is None:
            self.preprocess_doc_collection = PreprocessMultiFieldDocumentCollection.load(self.entity_collection_path)

    def init_all_sub_model(self):
        for model_dir_path, model_class, model_weight, load_model in self.sub_search_model_config:
            if not load_model:
                self.model_list.append(model_class.load_as_submodel(model_dir_path))
            else:
                model = model_class.load(model_dir_path)
                self.model_list.append(model)
                self.preprocess_doc_collection = model.preprocess_doc_collection
            self.model_weight_list.append(model_weight)

    def get_full_doc_score_vec(self, query):
        full_entity_score_vec = self.get_cache_score_vector(query)
        if full_entity_score_vec is not None:
            return full_entity_score_vec

        submodel_full_entity_score_vec_list = []
        for model in self.model_list:
            vector = model.get_full_doc_score_vec(query)
            submodel_full_entity_score_vec_list.append(vector)
        full_entity_score_vec = VectorUtil.get_weight_mean_vec(vector_list=submodel_full_entity_score_vec_list,
                                                               weight_list=self.model_weight_list)

        self.cache_entity_score_vector(query, full_entity_score_vec)
        return full_entity_score_vec

    def train_from_doc_collection_with_preprocessor(self, doc_collection, **config):
        """

        :param doc_collection:
        :param config: "search_model_class_list", each one is (model_path,model_class,model_weight)
        :return:
        """
        self.preprocess_doc_collection = doc_collection
        sub_search_model_config = []

        if "sub_search_model_config" in config.keys():
            sub_search_model_config = config["sub_search_model_config"]

        self.sub_search_model_config = sub_search_model_config
        if len(self.sub_search_model_config) == 0:
            raise Exception("search_model_class_config_list is []!!!")
        self.init_all_sub_model()
