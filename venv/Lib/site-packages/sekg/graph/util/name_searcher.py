import gensim

from sekg.graph.exporter.graph_data import GraphData, GraphDataReader, NodeInfoFactory


class KGNameSearcher(gensim.utils.SaveLoad):
    """
    search some node in KG by some keyword.
    """

    def __init__(self):
        self.full_name_2_id_set_map = {}
        self.id_2_full_name_set_map = {}
        self.keyword_2_id_set_map = {}
        self.id_2_keyword_set_map = {}

    def add_full_name_for_id(self, name, id):
        if not name:
            return
        if name not in self.full_name_2_id_set_map.keys():
            self.full_name_2_id_set_map[name] = set([])
        if id not in self.id_2_full_name_set_map.keys():
            self.id_2_full_name_set_map[id] = set([])

        self.full_name_2_id_set_map[name].add(id)
        self.id_2_full_name_set_map[id].add(name)

    def add_keyword_for_id(self, keyword, id):
        if not keyword:
            return
        if keyword not in self.keyword_2_id_set_map.keys():
            self.keyword_2_id_set_map[keyword] = set([])
        if id not in self.id_2_keyword_set_map.keys():
            self.id_2_keyword_set_map[id] = set([])

        self.keyword_2_id_set_map[keyword].add(id)

        self.id_2_keyword_set_map[id].add(keyword)

    def add_keyword_map_from_full_name(self, full_name, id):
        ## todo: change this to fix all

        full_name = full_name.split("(")[0]
        full_name = full_name.replace("-", " ").replace("(", " ").replace(")", " ").strip()

        self.add_keyword_for_id(full_name, id)

        name_words = full_name.split(" ")
        for word in name_words:
            self.add_keyword_for_id(word, id)
            self.add_keyword_for_id(word.lower(), id)

    @staticmethod
    def train_from_graph_data_file(graph_data_path, node_info_factory: NodeInfoFactory):
        """
        train the kg name searcher model from a graph data object
        :param node_info_factory: the nodeInfoFactory to create node from node json
        :param graph_data_path:
        :return:
        """
        graph_data: GraphData = GraphData.load(graph_data_path)

        searcher = KGNameSearcher()
        searcher.train(graph_data, node_info_factory)
        return searcher

    def train(self, graph_data: GraphData, node_info_factory: NodeInfoFactory):
        graph_data_reader = GraphDataReader(graph_data=graph_data, node_info_factory=node_info_factory)
        # todo: change the read all node from KG by iterate

        for id in graph_data.get_node_ids():
            node_info = graph_data_reader.get_node_info(id)
            name_list = node_info.get_all_names()
            for name in name_list:
                self.add_full_name_for_id(name, id)
                self.add_full_name_for_id(name.lower(), id)
                self.add_keyword_map_from_full_name(name, id)
                self.add_keyword_map_from_full_name(name.lower(), id)

    def search_by_full_name(self, full_name):
        if full_name in self.full_name_2_id_set_map.keys():
            return self.full_name_2_id_set_map[full_name]
        full_name = full_name.lower()
        if full_name in self.full_name_2_id_set_map.keys():
            return self.full_name_2_id_set_map[full_name]
        return set([])

    def search_by_keyword(self, word):
        if word in self.keyword_2_id_set_map.keys():
            return self.keyword_2_id_set_map[word]
        word = word.lower()

        if word in self.keyword_2_id_set_map.keys():
            return self.keyword_2_id_set_map[word]

        return set([])

    def clear(self):
        self.full_name_2_id_set_map = {}
        self.id_2_full_name_set_map = {}
        self.keyword_2_id_set_map = {}
        self.id_2_keyword_set_map = {}
