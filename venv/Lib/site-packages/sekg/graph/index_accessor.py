import traceback

from sekg.graph.accessor import GraphAccessor


class IndexGraphAccessor(GraphAccessor):
    __CYPHER_CREATE_INDEX = "CREATE INDEX ON :`{label}`(`{property_name}`)"
    __CYPHER_DROP_INDEX = "DROP INDEX ON :`{label}`(`{property_name}`)"

    def create_index(self, label, property_name):
        """
        create an index in one specific property with one label
        :param label: the label name to create index with
        :param property_name: the property name to index with
        :return:
        """
        query = self.__CYPHER_CREATE_INDEX.format(label=label, property_name=property_name)

        try:
            result = self.graph.evaluate(query)
            return result
        except Exception:
            traceback.print_exc()
            print("create index fail for %s" % query)
            return None

    def drop_index(self, label, property_name):
        """
        drop an index in one specific property with one label
        :param label: the label name to create index with
        :param property_name: the property name to index with
        :return:
        """
        query = self.__CYPHER_DROP_INDEX.format(label=label, property_name=property_name)

        try:
            result = self.graph.evaluate(query)
            return result
        except Exception:
            traceback.print_exc()
            print("drop index fail for %s" % query)
            return None
