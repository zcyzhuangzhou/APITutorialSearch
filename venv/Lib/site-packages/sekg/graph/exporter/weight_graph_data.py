from sekg.graph.exporter.graph_data import GraphData
from sekg.util.vector_util import VectorUtil

"""
this package is used to get a weight 
"""


class WeightGraphData:
    """
    this class is to get a weight  graph Data"""

    def __init__(self, graph_data):
        if isinstance(graph_data, GraphData):
            self.graph_data = graph_data
        else:
            self.graph_data = None

        self.relation_type_weight_dict = {}
        self.node_weight_dict = {}

    def precompute_weight(self):
        relation_count = self.graph_data.relation_num
        relation_type_count = {}

        node_degree_dict = {}

        for node_id in self.graph_data.get_node_ids():
            out_relation_list = self.graph_data.get_all_out_relation_dict_list(node_id=node_id)
            in_relation_list = self.graph_data.get_all_in_relation_dict_list(node_id=node_id)
            node_degree_dict[node_id] = len(out_relation_list) + len(in_relation_list)

            for relation_info_dict in out_relation_list:
                relation_type = relation_info_dict["relationType"]
                if relation_type not in relation_type_count.keys():
                    relation_type_count[relation_type] = 0
                relation_type_count[relation_type] = relation_type_count[relation_type] + 1

        self.relation_type_weight_dict = VectorUtil.compute_idf_weight_dict(total_num=relation_count,
                                                                            number_dict=relation_type_count)
        self.node_weight_dict = VectorUtil.compute_idf_weight_dict(total_num=relation_count * 2,
                                                                   number_dict=node_degree_dict)

    def get_node_weight(self, node_id):
        if node_id not in self.node_weight_dict.keys():
            return 1
        return self.node_weight_dict[node_id]

    def get_relation_type_weight(self, relation_type):
        if relation_type not in self.relation_type_weight_dict.keys():
            return 1
        return self.relation_type_weight_dict[relation_type]

    def get_relation_tuple_weight(self, start_node_id, end_node_id, relation_name):
        start_w = self.get_node_weight(start_node_id)
        end_w = self.get_node_weight(end_node_id)
        r_w = self.get_relation_type_weight(relation_name)

        return start_w * end_w * r_w
