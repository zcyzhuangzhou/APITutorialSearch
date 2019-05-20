import traceback

from py2neo import Subgraph

from sekg.graph.accessor import GraphAccessor


class MetadataGraphAccessor(GraphAccessor):
    """
    query the graph metadata
    """

    # todo: complete this accessor and pass test
    def get_max_id_for_node(self):
        """
            get the max id of Node in the whole graph
            :return: the max id
            """

        query = 'MATCH (n) return max(ID(n))'
        result = self.graph.evaluate(query)
        return result

    def get_max_id_for_relation(self):
        """
        get the max id of relation in the whole graph
        :return: the max id
        """
        query = 'MATCH ()-[n]-() return max(ID(n))'
        result = self.graph.evaluate(query)
        return result

    def get_node_num(self, label=None):
        # todo:
        """
        get the number of the nodes in the graph
        :return: the number of the nodes in the graph
        """
        if label == None:
            query = 'Match (n) RETURN count(n)'
        else:
            query = 'Match (n:`%s`) RETURN count(n)' % (label)
        return self.graph.evaluate(query)

    def get_relation_num(self):
        # todo:
        """
         get the number of the relations in the graph
        :return: the number of the relations in the graph
        """
        query = 'MATCH ()-->() RETURN count(*)'
        return self.graph.evaluate(query)

    def count_relation_type(self):
        """
        get the number of the relation type in the graph
        :return: the number of the relation type in the graph
        """
        query = "MATCH (n)-[r]->() RETURN type(r), count(*)"
        # query = 'CALL apoc.meta.stats() yield relTypeCount'
        result = self.graph.run(query)
        id_num = 0
        for record in result:
            id_num = id_num + 1
        return id_num

    def get_label_set(self):
        pass
        # todo:

    def get_relation_set(self):
        pass
        # todo:

    def expand_node(self, node_id, limit=40):
        """
        get the directly_adjacent_nodes of one node
        :return: return value is a subgraph
        """
        low_quality_query = "Match (n:entity)-[r]-(m:entity) where ID(n)={start_id} return distinct r,n,m limit {limit}"
        low_quality_query = low_quality_query.format(start_id=node_id, limit=limit)
        try:
            nodes = []
            relationships = []
            record_list_for_all_relation = self.graph.run(low_quality_query)

            for record in record_list_for_all_relation:
                r = record["r"]
                relationships.append(r)
                nodes.append(record["n"])
                nodes.append(record["m"])

            if nodes:
                return Subgraph(nodes, relationships)
            else:
                return None
        except Exception:
            traceback.print_exc()
            return None
