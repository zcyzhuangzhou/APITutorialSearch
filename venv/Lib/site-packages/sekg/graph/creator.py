from py2neo import Node


class NodeBuilder:
    """
    a builder for Node
    """

    def __init__(self):
        self.labels = []
        self.property_dict = {}

    def add_labels(self, *labels):
        """
        add labels for Node
        :param labels: all labels need to be added
        :return: a NodeBuilder object
        """
        self.labels.extend(labels)
        self.labels = list(set(self.labels))

        return self

    def add_label(self, label):
        """
        add a label for Node
        :param label: label, string
        :return: a NodeBuilder object
        """
        if label not in self.labels:
            self.labels.append(label)
        return self

    def add_property(self, **property_dict):
        self.property_dict = dict(self.property_dict, **property_dict)
        return self

    def add_one_property(self, property_name, property_value):
        if property_value is None or property_value is "":
            return self
        self.property_dict[property_name] = property_value
        return self

    def add_entity_label(self):
        return self.add_labels('entity')

    def build(self):
        node = Node(*self.labels)
        for key in self.property_dict:
            node[key] = self.property_dict[key]
        return node

    def get_labels(self):
        """
        get the labels for current built node
        :return: a set of labels
        """
        return self.labels

    def get_properties(self):
        """
        get the properties for current built node
        :return: a dict of properties
        """
        return self.property_dict

    def build_node_json(self):
        node_json = {"id": -1, "properties": self.property_dict, "labels": self.labels}
        return node_json
