import gensim

from sekg.ir.preprocessor.base import Preprocessor


class MultiFieldDocumentCollection(gensim.utils.SaveLoad):
    """
    This class is a wrapper for multi field document collection. It contain many MultiFieldDocument instance. Each one stand for a document.
    """

    def __init__(self, documents=None):
        self.documents = []
        self.field_set = set([])
        if documents:
            for document in documents:
                self.add_document(document)

        self.doc_id_2_documents_map = {}
        self.doc_id_2_doc_index_map = {}

    def get_num(self):
        return len(self.documents)

    def add_document(self, document):
        """
        :param document:
        :return: False, the doc with the id already exist. True, add new doc success
        """
        doc_id = document.id
        if doc_id in self.doc_id_2_documents_map.keys():
            return False
        self.documents.append(document)
        self.field_set.update(document.get_field_set())
        self.doc_id_2_documents_map[doc_id] = document
        self.doc_id_2_doc_index_map[doc_id] = len(self.documents) - 1
        return True

    def get_by_id(self, id):
        if id in self.doc_id_2_documents_map.keys():
            return self.doc_id_2_documents_map[id]
        return None

    def get_by_index(self, index):
        return self.documents[index]

    def __str__(self):
        return "Documents(Num=%d)" % (self.get_num())

    def clear(self):
        self.documents = []
        self.field_set = set([])

    def get_field_set(self):
        return self.field_set

    def get_document_list(self):
        return self.documents

    def add_field_to_doc(self, doc_id, field_name, value):
        doc = self.get_by_id(id=doc_id)
        if doc is None:
            return
        doc.add_field(field_name, value)

    def doc_index_to_doc_id(self, index):
        return self.get_by_index(index).get_document_id()

    def doc_id_to_doc_index(self, doc_id):
        return self.doc_id_2_doc_index_map[doc_id]


class PreprocessMultiFieldDocumentCollection(gensim.utils.SaveLoad):
    """
        This class is a wrapper for multi field document collection. It contain many MultiFieldDocument instance. Each one stand for a document.
        But it contain preproccessed doc and the original doc.
    """

    def __init__(self, preprocessor: Preprocessor):
        self.raw_doc_collection = MultiFieldDocumentCollection()
        self.preprocess_doc_collection = MultiFieldDocumentCollection()

        if preprocessor is None:
            raise Exception("preprocessor is None")

        if isinstance(preprocessor, Preprocessor):
            self.preprocessor = preprocessor
        else:
            raise Exception("preprocessor is not a Preprocessor instance")

    @staticmethod
    def create_from_doc_collection(preprocessor: Preprocessor, doc_collection: MultiFieldDocumentCollection):
        """
        create the preprocessed document collection
        :param preprocessor: preprocessor use to preprocess the document
        :param doc_collection: the document collection need to preprocess
        :return: PreprocessMultiFieldDocumentCollection
        """
        pre_doc_collection = PreprocessMultiFieldDocumentCollection(preprocessor)
        pre_doc_collection.add_raw_doc_collection(doc_collection)
        return pre_doc_collection

    def get_field_set(self):
        return self.raw_doc_collection.get_field_set()

    def get_preprocessor(self):
        return self.preprocessor

    def get_num(self):
        return self.raw_doc_collection.get_num()

    def add_raw_doc_and_preprocess_doc_pair(self, raw_doc, preprocess_doc):
        self.raw_doc_collection.add_document(raw_doc)
        self.preprocess_doc_collection.add_document(preprocess_doc)

    def get_raw_doc_and_preprocess_doc_pair(self, index):
        return (
            self.get_raw_doc_by_index(index),
            self.get_preprocess_doc_by_index(index)
        )

    def get_raw_doc_by_index(self, index):
        """
        get the
        :param index:
        :return:
        """
        return self.raw_doc_collection.get_by_index(index)

    def get_raw_doc_by_id(self, id):
        return self.raw_doc_collection.get_by_id(id)

    def get_preprocess_doc_by_id(self, id):
        return self.preprocess_doc_collection.get_by_id(id)

    def get_preprocess_doc_by_index(self, index):
        """
        the preprocess doc is list of str
        :param index:
        :return: a list of str
        """
        return self.preprocess_doc_collection.get_by_index(index)

    def __str__(self):
        return "Documents(Num=%d)" % (self.get_num())

    def add_raw_doc(self, raw_doc):
        if isinstance(raw_doc, MultiFieldDocument):
            self.raw_doc_collection.add_document(raw_doc)

            preprocess_doc = MultiFieldDocument(id=raw_doc.get_document_id(), name=raw_doc.get_name())
            for field, field_doc in raw_doc.get_all_field_doc_map().items():
                preprocess_field_doc_words = self.preprocessor.clean(field_doc)
                preprocess_doc.add_field(field_name=field, field_document=" ".join(preprocess_field_doc_words))
            self.preprocess_doc_collection.add_document(preprocess_doc)

    def add_raw_doc_collection(self, raw_doc_collection):
        for index, raw_doc in enumerate(raw_doc_collection.get_document_list()):
            self.add_raw_doc(raw_doc)

    def clear(self):
        self.raw_doc_collection.clear()
        self.preprocess_doc_collection.clear()

    def get_all_preprocess_documents_text(self):
        """
        get list of str
        :return:
        """
        result = []
        for preprocess_doc in self.preprocess_doc_collection.documents:
            result.append(preprocess_doc.get_document_text())
        return result

    def get_all_preprocess_documents_text_words(self):
        """
        get list of str
        :return:
        """
        result = []
        for preprocess_doc in self.preprocess_doc_collection.documents:
            result.append(preprocess_doc.get_document_text_words())
        return result

    def get_all_preprocess_document_list(self):
        """
        get the list of all preprocess_document( containing multifield)
        :return:
        """
        return self.preprocess_doc_collection.get_document_list()

    def get_all_preprocess_field_words(self, field_name):
        """
        get list of str
        :return:
        """
        result = []
        for preprocess_doc in self.preprocess_doc_collection.documents:
            result.append(preprocess_doc.get_doc_words_by_field(field_name))
        return result

    def get_by_index(self, index):
        return self.raw_doc_collection.get_by_index(index)

    def doc_index_to_doc_id(self, index):
        return self.raw_doc_collection.doc_index_to_doc_id(index)

    def doc_id_to_doc_index(self, doc_id):
        return self.raw_doc_collection.doc_id_to_doc_index(doc_id)


## todo: try to reduce the MultiFieldDocument and PreprocessorMultiFieldDocument

class MultiFieldDocument:
    """
    This class is a wrapper for the document with multi field.
    For example, for a wikipedia doc. we can has two field "title","body".
    for a stack overflow question, we could have "question title","tags","question body","accept answer".
    For this purpose, this class is used to wrapper the doc like this has multi field.

    """

    def __init__(self, id, name, field_doc=None):
        if field_doc is None:
            field_doc = {}
        self.id = id
        self.name = name
        self.field_doc = field_doc

    def add_field(self, field_name, field_document):
        self.field_doc[field_name] = field_document

    def get_field_set(self):
        return self.field_doc.keys()

    def get_doc_text_by_field(self, field_name):
        """

        :param field_name:
        :return:
        """
        if field_name in self.field_doc.keys():
            return self.field_doc[field_name]
        else:
            return ""

    def get_doc_words_by_field(self, field_name):
        """
        the return result could be list of str or a text
        :param field_name:
        :return:
        """
        text = self.get_doc_text_by_field(field_name)
        return text.split()

    def get_all_field_doc_map(self):
        return self.field_doc

    def get_name(self):
        return self.name

    def get_document_id(self):
        return self.id

    def get_document_text(self):
        """
        get all the text from this MultiFieldDocument by conbining text from all field.
        :return: return a str
        """
        docs = []
        for field_name in self.get_field_set():
            doc = self.get_doc_text_by_field(field_name=field_name)
            docs.append(doc)
        return "\n".join(docs)

    def get_document_text_words(self):
        """
        get all the text from this MultiFieldDocument by conbining text from all field.
        :return: return a iteration of str. each one is a word in doc field.
        """

        docs = []
        for field_name in self.get_field_set():
            doc = self.get_doc_words_by_field(field_name=field_name)
            docs.extend(doc)
        return docs

    def __repr__(self):
        return "<MultiFieldDocument= id=%d name=%r doc=%r>" % (self.get_document_id(), self.get_name(), self.field_doc)
