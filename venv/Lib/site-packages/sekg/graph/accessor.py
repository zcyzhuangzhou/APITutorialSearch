#!/usr/bin/python
# -*- coding: UTF-8 -*-
import time

from py2neo import Graph, Node, Relationship

__CYPHER_MERGE_RELATION__ = 'MATCH (a) WHERE id(a) = {start_id} MATCH (b) WHERE id(b) ={end_id} MERGE (a)-[r:`{relation_name}`]->(b) RETURN r'


class GraphAccessor:
    """
    this class is for do some query or operation on neo4j graph, is wrapper of py2neo.Graph Object,
    You can get init it with a py2neo.Graph obejct and get the py2neo.Graph object from the GraphAccessor instance
    """

    def __init__(self, graph=None):
        if isinstance(graph, GraphAccessor):
            self._graph = graph.graph
        elif isinstance(graph, Graph):
            self._graph = graph
        else:
            self._graph = None

    @property
    def graph(self):
        return self._graph

    @graph.setter
    def graph(self, graph):
        if isinstance(graph, GraphAccessor):
            self._graph = graph.graph
        elif isinstance(graph, Graph):
            self._graph = graph
        else:
            self._graph = None

    @graph.deleter
    def graph(self):
        del self._graph

    def is_connect(self):
        """
        check if the GraphAccessor valid
        :return: True, valid; False, not valid
        """
        if self._graph is None:
            return False
        else:
            return True

    @staticmethod
    def get_id(graph_element):
        """
        get the id of one node or relationship in graph
        :return: the unique id of the node or relationship,None is stand for it is not exist in remote neo4j
        """
        if isinstance(graph_element, Node) or isinstance(graph_element, Relationship):
            return graph_element.identity
        else:
            return None

    @staticmethod
    def is_remote(graph_element):
        """
        get the id of one node or relationship in graph
        :return: the unique id of the node or relationship,None is stand for it is not exist in remote neo4j
        """
        if isinstance(graph_element, Node) or isinstance(graph_element, Relationship):
            if (graph_element.identity is None):
                return False
            else:
                return True
        else:
            return False

    def create_relation_without_duplicate(self, start_node, relation_str, end_node):
        """
        merge a relation to graph, and update the create. And will create log and update some metadata.
        :param start_node: the start node must be exist in remote graph
        :param relation_str: the relation type string
        :param end_node: the end node must be exist in remote graph
        :return:
        """
        if start_node is None:
            print("fail create relation for start node is None")
            return None
        if end_node is None:
            print("fail create relation for end node is None")
            return None
        if not relation_str:
            print("fail create relation for empty relation string")
            return None
        if not self.is_remote(start_node):
            print("fail create relation for start node is not remote")
            return None
        if not self.is_remote(end_node):
            print("fail create relation for start node is not remote")
            return None

        r = None
        try:
            query = __CYPHER_MERGE_RELATION__.format(
                relation_name=relation_str, start_id=start_node.identity, end_id=end_node.identity)
            r = self.graph.evaluate(query)
        except Exception as error:
            print(error)

        if r is None:
            print("merge relation fail")
            return None
        self.update_metadata(r)
        return r

    def create_or_update_node(self, node, primary_label, primary_property):

        """
        merge a node to graph, and update the create. And will create log and update some metadata.
        :param primary_property: the merge primary key
        :param primary_label: the merge primary property key
        :param node: the node need to merged
        :return: the node created
        """
        if node is None:
            print("fail create node for start node is None")
            return None
        if not isinstance(node, Node):
            print("the node is not a node object")
            return None

        primary_property_value = node[primary_property]
        if primary_property_value is None:
            print("the primary_property_value {property} is not exist in node object".format(property=primary_property))

            return None

        remote_node = self.find_node(primary_label=primary_label, primary_property=primary_property,
                                     primary_property_value=primary_property_value)
        if remote_node is None:
            remote_node = self.create_node(node)
        else:
            remote_node = self.update_remote_node_by_node(remote_node=remote_node, source_node=node)

        return remote_node

    def update_metadata(self, graph_element):
        if not self.is_remote(graph_element):
            print("fail update metadata because the graph_element is not remote")
            return False

        modify_time = int(time.strftime('%Y%m%d%H%M%S', time.localtime(time.time())))
        if "_create_time" not in graph_element:
            graph_element["_create_time"] = modify_time
        if "_update_time" not in graph_element:
            graph_element["_update_time"] = modify_time
        else:
            graph_element["_update_time"] = modify_time
        if "_modify_version" not in graph_element:
            graph_element["_modify_version"] = 1
        else:
            graph_element["_modify_version"] = graph_element["_modify_version"] + 1
        print("merge id=%r relation version:%d data:%r", graph_element.identity,
              graph_element["_modify_version"], graph_element)
        self.graph.push(graph_element)
        return True

    def update_remote_node(self, remote_node, property_dict, labels=None):
        """
         update the remote node by copy all properties and labels to remote_node,
         the properties in remote_node but not in property_dict remind unchanged.
        :param property_dict: provide the properties update the remote node
        :param labels: provide the labels to update the remote node, if [], stand for not change the labels of remote
        :param remote_node: the node in remote graph instance
        :return: the remote node updated, None if fail
        """

        if remote_node is None:
            print("fail push node for node is None")
            return None
        if not isinstance(remote_node, Node):
            print("fail the remote_node is not a node object")

        if not self.is_remote(remote_node):
            print("fail push node for node is not a remote node %r", remote_node)
            return

        for k, v in property_dict.items():
            remote_node[k] = v

        self.graph.push(remote_node)
        self.update_labels(remote_node, labels)

        return remote_node

    def update_remote_node_by_node(self, source_node, remote_node):
        """
        update the remote node by copy all property and labels from source node
        :param remote_node: the node in remote graph instance
        :param source_node: provide the properties and labels to update the remote node
        :return: the remote node updated, None if fail
        """
        if source_node is None:
            print("fail source_node is None")
            return None
        if not isinstance(source_node, Node):
            print("fail the source_node is not a node object")

        property_dict = dict(source_node)
        return self.update_remote_node(remote_node=remote_node, property_dict=property_dict, labels=source_node.labels)

    def find_node_by_id(self, node_id):
        """
        get a node by id
        :param node_id: the id of node
        :return: the Node object
        """
        return self.graph.nodes.get(node_id)

    def find_relation_by_id(self, node_id):
        """
        get a node by id
        :param node_id: the id of node
        :return: the Node object
        """
        return self.graph.relationships.get(node_id)

    def find_node(self, primary_label, primary_property, primary_property_value):
        """
        find a unique node by primary label, primary property, primary property value.
        eg. (primary label="api class",primary property="api_id",primary property value=3)
        :param primary_label: the match primary label.
        :param primary_property: the match primary property name
        :param primary_property_value: the match primary property value
        :return: the node found, None if the node not exist
        """
        try:
            query = 'MATCH (n:`%s`{`%s`:%r}) return n' % (primary_label, primary_property, primary_property_value)
            result = self.graph.evaluate(query)
            return result
        except Exception as error:
            print(error)
            return None

    def create_node(self, node):
        """
        create a new node in kg, and support the name of relation and labels,property name could be
        :param node: the node need to be create
        :return: the created node
        """
        if node is None:
            print("fail create node for node is None")
            return None
        if not isinstance(node, Node):
            print("fail the node is not a node object")

        if self.is_remote(node):
            print("fail create node because it is a remote node %s", str(node))
            return

        try:
            labels_str = ":".join(["`" + str(label) + "`" for label in node.labels])

            properties_str = ",".join(["`%s`:%r" % (property, value) for property, value in node.items()])

            query = 'create (n:%s {%s}) return n' % (labels_str, properties_str)
            node = self.graph.evaluate(query)
            self.update_metadata(node)
        except Exception as error:
            print("fail create node when execute the cypher")
            print(error)
            return None

        return node

    def update_labels(self, remote_node, labels):

        if labels is None:
            labels = set([])
        if not labels:
            return remote_node

        labels_added = labels - remote_node.labels
        labels_delete = remote_node.labels - labels
        try:
            if labels_added:
                labels_added_str = ":".join(["`" + str(label) + "`" for label in labels_added])
                set_str = "SET n:%s" % labels_added_str
            else:
                set_str = ""

            if labels_delete:
                labels_delete_str = ":".join(["`" + str(label) + "`" for label in labels_delete])

                remove_str = "REMOVE n:%s" % labels_delete_str
            else:
                remove_str = ""

            query = 'match (n) where id(n)={node_id}  {remove_str}  {set_str} return n'.format(
                node_id=remote_node.identity, remove_str=remove_str, set_str=set_str)
            remote_node = self.graph.evaluate(query)
            self.update_metadata(remote_node)
            return remote_node
        except Exception as error:
            print("fail create node when execute the cypher")
            print(error)
            return remote_node

    def get_relation_type_string(self, relation):
        if isinstance(relation, Relationship):
            return type(relation).__name__
        else:
            return None
