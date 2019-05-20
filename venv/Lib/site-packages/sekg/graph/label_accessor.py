import traceback

from sekg.graph.accessor import GraphAccessor


class LabelGraphAccessor(GraphAccessor):

    def add_label_for_one_label_node(self, match_node_label, added_label):
        return self.add_labels(match_node_label_list=[match_node_label], added_label_list=[added_label])

    def delete_label_for_one_label_node(self, node_label_matched, delete_label):
        return self.delete_labels(node_labels_matched=[node_label_matched], labels_delete=[delete_label])

    def add_labels(self, match_node_label_list, added_label_list):
        labels_added_str = ":".join(["`" + str(label) + "`" for label in added_label_list])
        set_str = "SET n:%s" % labels_added_str

        labels_matched_str = ""
        for label in match_node_label_list:
            labels_matched_str += ":"
            labels_matched_str += "`"
            labels_matched_str += label
            labels_matched_str += "`"

        query = 'match (n{match_labels}) {set_str} return count(n)'.format(
            match_labels=labels_matched_str, set_str=set_str)

        try:
            return self.graph.evaluate(query)
        except Exception as error:
            traceback.print_exc()
            print("fail for query=%s" % query)
            return None

    def delete_labels(self, node_labels_matched, labels_delete):
        labels_delete_str = ":".join(["`" + str(label) + "`" for label in labels_delete])
        remove_str = "REMOVE n:%s" % labels_delete_str
        labels_matched_str = ""
        for label in node_labels_matched:
            labels_matched_str += ":"
            labels_matched_str += "`"
            labels_matched_str += label
            labels_matched_str += "`"

        query = 'match (n{match_labels}) {remove_str} return count(n)'.format(
            match_labels=labels_matched_str, remove_str=remove_str)
        try:
            return self.graph.evaluate(query)
        except Exception as error:
            traceback.print_exc()
            print("fail for query=%s" % query)
            return None
