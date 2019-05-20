#!/usr/bin/env python
# -*- coding: utf-8 -*-

import networkx as nx
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

from sekg.graph.exporter.weight_graph_data import WeightGraphData
from sekg.model.node2vec import node2vec


class GraphNode2VecTrainer():
    def __init__(self, graph_data):
        self.graph_data = graph_data
        self.node_num = graph_data.get_node_num()
        self.nx_G_instance = None

    def init_graph(self, weight=False):
        if weight == False:
            self.init_unweight_graph()
        else:
            self.init_weight_graph()

    def init_unweight_graph(self):
        node_ids = self.graph_data.get_node_ids()
        relation_pairs = self.graph_data.get_relation_pairs()
        print("node num=%d" % self.node_num)
        print("relation num=%d" % len(relation_pairs))
        G = nx.DiGraph()
        G.add_nodes_from(node_ids)
        G.add_edges_from(relation_pairs, weight=1.0)
        # todo: a relation weight support
        self.nx_G_instance = G
        print("init graph trainer by unweight relations")

    def init_weight_graph(self):

        node_ids = self.graph_data.get_node_ids()
        relation_pairs_with_type = self.graph_data.get_relation_pairs_with_type()
        print("node num=%d" % self.node_num)
        print("relation num=%d" % len(relation_pairs_with_type))

        print(" pre-compute weight start")

        weight_graph_data = WeightGraphData(self.graph_data)
        weight_graph_data.precompute_weight()
        print("pre-compute weight end")

        weight_relation_tuples = []
        for (start_id, relation_type, end_id) in relation_pairs_with_type:
            weight = weight_graph_data.get_relation_tuple_weight(start_node_id=start_id, relation_name=relation_type,
                                                                 end_node_id=end_id)
            weight_relation_tuples.append((start_id, end_id, weight))

        G = nx.DiGraph()
        G.add_nodes_from(node_ids)
        G.add_weighted_edges_from(weight_relation_tuples)
        self.nx_G_instance = G
        print("init graph trainer by weighted relations(tf-idf tuple)")

    def generate_random_path(self, rw_path_store_path, directed=False, p=1, q=1, num_walks=10, walk_length=80,
                             ):
        print("start generate graph random path")

        G = node2vec.Graph(self.nx_G_instance, directed, p, q)
        G.preprocess_transition_probs()
        walks = G.simulate_walks(num_walks, walk_length)
        # todo: may be load all into memory has problem?
        with open(rw_path_store_path, 'w') as write_f:
            for walk in walks:
                path_str = " ".join([str(item) for item in walk])
                write_f.writelines('%s\n' % (path_str))

        print("complete generate graph random path")

    @staticmethod
    def train(rw_path_store_path, model_path, dimensions=100,
              workers=4):
        """
        train the graph vector from rw_path
        :param rw_path_store_path: the random walk for one graph
        :param model_path: the output word2vec model path
        :param dimensions: the dimensions of word2vec
        :param workers: the num of pipeline training
        :return:
        """
        print("save graph2vec training")

        # Learn embeddings by optimizing the Skipgram objective using SGD.
        w2v = Word2Vec(LineSentence(rw_path_store_path), size=dimensions, min_count=0, sg=1, workers=workers)
        w2v.wv.save(model_path)
        print("save graph2vec to %s" % model_path)
        return w2v
