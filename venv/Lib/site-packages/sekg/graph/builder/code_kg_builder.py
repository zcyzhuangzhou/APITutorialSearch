from sekg.constant.code import CodeEntityCategory, CodeEntityRelationCategory
from sekg.graph.creator import NodeBuilder
from sekg.graph.exporter.graph_data import GraphData


class CodeElementGraphDataBuilder:
    VALID_IDENTITY_CHAR_SET = " ?,.QWERTYUIOPASDFGHJKLZXCVBNMqwertyuiopasdfghjklzxcvbnm1234567890_()<>[]"

    def __init__(self, graph_data):
        self.graph_data = graph_data
        self.init_indexs()

    def init_indexs(self):
        self.graph_data.create_index_on_property("qualified_name", "short_description")

    def format_qualified_name(self, name):
        name = name.replace(", ", ",")
        for c in name:
            if c not in self.VALID_IDENTITY_CHAR_SET:
                return ""
        return name

    def parse_construct_to_javaparser_style(self, name):
        if "(" not in name:
            return name
        head = name.split("(")[0]
        tail = name.split("(")[1]

        return "{class_part}.<init>({parameter}".format(class_part=self.get_parent_name_for_api(head), parameter=tail)

    @staticmethod
    def get_simple_name_for_type(name):
        return name.split(".")[-1]

    @staticmethod
    def get_parent_name_for_api(name):
        split = name.split("(")[0].split(".")
        return ".".join(split[:-1])

    @staticmethod
    def get_parameter_num(method_name):

        if method_name.endswith("()"):
            return 0
        return method_name.count(",") + 1

    def find_match_method_by_actual_method_call(self, actual_method_call, method_nodes):
        if len(method_nodes) == 0:
            return None
        if len(method_nodes) == 1:
            return method_nodes[0]

        actual_parameter = self.get_parameter_num(actual_method_call)
        parameter_num_2_methods = {}

        for candidate_node in method_nodes:
            team = self.get_parameter_num(candidate_node["properties"]["qualified_name"])
            if team not in parameter_num_2_methods:
                parameter_num_2_methods[team] = []
            parameter_num_2_methods[team].append(candidate_node)

        if actual_parameter not in parameter_num_2_methods:
            print("match method fail for %r following candidate %r" % (
                actual_method_call,
                [candidate_node["properties"]["qualified_name"] for candidate_node in method_nodes]))

            return None

        if len(parameter_num_2_methods[actual_parameter]) > 1:
            print("match method fail for %r following candidate %r" % (
                actual_method_call,
                [candidate_node["properties"]["qualified_name"] for candidate_node in method_nodes]))

            return None

        return parameter_num_2_methods[actual_parameter][0]

    def add_base_value_entity_node(self, value_type, value_name, short_description="",
                                   entity_category=CodeEntityCategory.CATEGORY_VALUE, **extra_properties):
        qualified_name = "{type} {var_name}".format(type=value_type,
                                                    var_name=value_name)
        simple_name = "{type} {var_name}".format(type=self.get_simple_name_for_type(value_type),
                                                 var_name=value_name)

        code_element = {
            "qualified_name": qualified_name,
            "simple_name": simple_name,
            "type": value_type,
            "value_name": value_name,
            "short_description": short_description,
        }
        for k, v in extra_properties.items():
            if k not in code_element:
                code_element[k] = v
        cat_labels = CodeEntityCategory.to_str_list(entity_category)

        builder = NodeBuilder().add_property(**code_element).add_entity_label().add_labels("code_element",
                                                                                           *cat_labels)

        new_field_node_id = self.graph_data.add_node_with_multi_primary_property(
            node_id=GraphData.UNASSIGNED_NODE_ID,
            node_labels=builder.get_labels(),
            node_properties=builder.get_properties(),
            primary_property_names=["qualified_name", "short_description"])

        type_node_id = self.add_type_node(value_type)

        self.graph_data.add_relation(new_field_node_id,
                                     CodeEntityRelationCategory.to_str(
                                         CodeEntityRelationCategory.RELATION_CATEGORY_TYPE_OF),
                                     type_node_id)

        return new_field_node_id

    def add_type_node(self, type_str):
        """
        add a new node stand for a type in GraphData.
        :param type_str:
        :return:
        """

        cat_label = CodeEntityCategory.to_str(CodeEntityCategory.CATEGORY_CLASS)
        builder = NodeBuilder().add_entity_label().add_labels("code_element", cat_label).add_one_property(
            "qualified_name",
            type_str)
        exist_json = self.graph_data.find_one_node_by_property(property_name="qualified_name",
                                                               property_value=type_str)

        if exist_json:
            return exist_json[self.graph_data.DEFAULT_KEY_NODE_ID]

        # print("add new type %s" % (type_str))

        type_node_id = self.graph_data.add_node(
            node_id=GraphData.UNASSIGNED_NODE_ID,
            node_labels=builder.get_labels(),
            node_properties=builder.get_properties(),
            primary_property_name="qualified_name")

        if type_node_id == GraphData.UNASSIGNED_NODE_ID:
            print("add new type fail: %r- %r" % (type_node_id, type_str))

        if "[]" in type_str:
            base_type_for_array = type_str.strip("[]")
            # print("array type- %s, base type- %s" % (type_str, base_type_for_array))
            base_type_node_id = self.add_type_node(base_type_for_array)

            self.graph_data.add_relation(type_node_id, "array of", base_type_node_id)

        return type_node_id

    def add_base_overrloading_method_node(self, simple_method_name):
        """
        add a new node stand for a base abstract method node. etc. java.lang.Math.abs, the method without parameters
        :param simple_method_name:  java.lang.Math.abs, the method without parameters
        :return:
        """

        cat_label = CodeEntityCategory.to_str(CodeEntityCategory.CATEGORY_BASE_OVERRIDE_METHOD)

        builder = NodeBuilder().add_entity_label().add_labels("code_element", cat_label).add_one_property(
            "qualified_name",
            simple_method_name)
        exist_json = self.graph_data.find_one_node_by_property(property_name="qualified_name",
                                                               property_value=simple_method_name)

        if exist_json:
            return exist_json[self.graph_data.DEFAULT_KEY_NODE_ID]

        new_node_id = self.graph_data.add_node(
            node_id=GraphData.UNASSIGNED_NODE_ID,
            node_labels=builder.get_labels(),
            node_properties=builder.get_properties(),
            primary_property_name="qualified_name")

        return new_node_id

    def add_normal_code_element_entity(self, qualified_name, entity_category, **extra_properties):
        """
        add the normal code element whose their qualified_name are unique
        :param qualified_name: the qualified_name of the code element
        :param entity_category:
        :param extra_properties:
        :return:
        """
        format_qualified_name = self.format_qualified_name(qualified_name)
        if not format_qualified_name:
            return GraphData.UNASSIGNED_NODE_ID

        code_element = {
            "qualified_name": format_qualified_name,
            "entity_category": entity_category
        }

        for k, v in extra_properties.items():
            if k not in code_element:
                code_element[k] = v

        cate_labels = CodeEntityCategory.to_str_list(entity_category)

        builder = NodeBuilder().add_entity_label().add_labels("code_element", *cate_labels).add_property(**code_element)

        node_id = self.graph_data.add_node(
            node_id=GraphData.UNASSIGNED_NODE_ID,
            node_labels=builder.get_labels(),
            node_properties=builder.get_properties(),
            primary_property_name="qualified_name")
        return node_id

    def add_method_use_class_relation(self, start_method_name, end_class_name):
        if not start_method_name or not end_class_name:
            print("start name or end name is None")
            return False
        raw_start_name = start_method_name

        relation_type_str = CodeEntityRelationCategory.to_str(
            CodeEntityRelationCategory.RELATION_CATEGORY_METHOD_IMPLEMENT_CODE_USE_CLASS)

        start_name = self.format_qualified_name(raw_start_name)
        start_node = self.graph_data.find_one_node_by_property(property_name="qualified_name",
                                                               property_value=start_name)

        if start_node is None:
            print("fail because of start node raw=%r format=%r" % (raw_start_name, start_name))
            return False
        type_node_id = self.add_type_node(end_class_name)

        self.graph_data.add_relation(start_node[GraphData.DEFAULT_KEY_NODE_ID],
                                     relation_type_str,
                                     type_node_id)

        return True

    def add_method_call_relation(self, start_name, end_name):
        if not start_name or not end_name:
            print("start name or end name is None")
            return False
        raw_start_name = start_name

        relation_type_str = CodeEntityRelationCategory.to_str(
            CodeEntityRelationCategory.RELATION_CATEGORY_METHOD_IMPLEMENT_CODE_CALL_METHOD)

        start_name = self.format_qualified_name(raw_start_name)
        start_node = self.graph_data.find_one_node_by_property(property_name="qualified_name",
                                                               property_value=start_name)

        if start_node is None:
            print("fail because of start node raw=%r format=%r" % (raw_start_name, start_name))
            return False

        end_node = self.graph_data.find_one_node_by_property(property_name="qualified_name",
                                                             property_value=end_name)

        if end_node is not None:
            self.graph_data.add_relation(start_node[GraphData.DEFAULT_KEY_NODE_ID],
                                         relation_type_str,
                                         end_node[GraphData.DEFAULT_KEY_NODE_ID])
            return True

        simple_name = end_name.split("(")[0] + "("
        candidate_nodes = self.graph_data.find_nodes_by_property_value_starts_with(
            property_name="qualified_name",
            property_value_starter=simple_name)
        end_node = self.find_match_method_by_actual_method_call(end_name, candidate_nodes)

        if end_node is None:
            end_abstract_method_node_id = self.add_base_overrloading_method_node(end_name.split("(")[0])
        else:
            end_abstract_method_node_id = end_node[GraphData.DEFAULT_KEY_NODE_ID]

        self.graph_data.add_relation(start_node[GraphData.DEFAULT_KEY_NODE_ID],
                                     relation_type_str,
                                     end_abstract_method_node_id)

        return True

    def build_abstract_overloading_relation(self):
        """
        for all methods, add a abstract new node stand for the overloading method with different parameters.
        its qualified name is simple method name without parameters. eg. java.lang.Math.abs
        :return:
        """
        print("start to build abstract_overloading_relation")
        print(self.graph_data)
        self.graph_data.print_graph_info()

        all_node_id_set = self.graph_data.get_node_ids()
        finish_node_id_set = set([])

        METHOD_CATE_STR = CodeEntityCategory.to_str(CodeEntityCategory.CATEGORY_METHOD)
        # todo: change to find node by label
        for node_id in all_node_id_set:
            if node_id in finish_node_id_set:
                continue

            node_json = self.graph_data.get_node_info_dict(node_id=node_id)
            if METHOD_CATE_STR not in node_json[GraphData.DEFAULT_KEY_NODE_LABELS]:
                finish_node_id_set.add(node_id)
                continue

            method_qualified_name = node_json[GraphData.DEFAULT_KEY_NODE_PROPERTIES]["qualified_name"]
            if "(" not in method_qualified_name:
                finish_node_id_set.add(node_id)
                continue

            base_overloading_method_simple_name = method_qualified_name.split("(")[0]
            new_base_method_node_id = self.add_base_overrloading_method_node(
                base_overloading_method_simple_name)

            finish_node_id_set.add(new_base_method_node_id)

            candidate_nodes = self.graph_data.find_nodes_by_property_value_starts_with(
                property_name="qualified_name",
                property_value_starter=base_overloading_method_simple_name + "(")

            for candidate_node in candidate_nodes:
                team_method_node_id = candidate_node[GraphData.DEFAULT_KEY_NODE_ID]
                finish_node_id_set.add(team_method_node_id)
                if team_method_node_id == new_base_method_node_id:
                    continue

                self.graph_data.add_relation(team_method_node_id,
                                             CodeEntityRelationCategory.to_str(
                                                 CodeEntityRelationCategory.RELATION_CATEGORY_METHOD_OVERLOADING),
                                             new_base_method_node_id)
        print(self.graph_data)
        self.graph_data.print_graph_info()
        print("end import abstract overloading_relation entity json")

    def build_belong_to_relation(self):
        """
        for all methods, add a abstract new node stand for the overloading method with different parameters.
        its qualified name is simple method name without parameters. eg. java.lang.Math.abs
        :return:
        """
        print("start build belong to relation")
        print(self.graph_data)
        self.graph_data.print_graph_info()

        all_node_id_set = self.graph_data.get_node_ids()
        finish_node_id_set = set([])

        VALID_LABELS = {
            CodeEntityCategory.to_str(CodeEntityCategory.CATEGORY_PACKAGE),
            CodeEntityCategory.to_str(CodeEntityCategory.CATEGORY_CLASS),
            CodeEntityCategory.to_str(CodeEntityCategory.CATEGORY_INTERFACE),
            CodeEntityCategory.to_str(CodeEntityCategory.CATEGORY_METHOD),
            CodeEntityCategory.to_str(CodeEntityCategory.CATEGORY_ENUM_CLASS),
            CodeEntityCategory.to_str(CodeEntityCategory.CATEGORY_BASE_OVERRIDE_METHOD),
            CodeEntityCategory.to_str(CodeEntityCategory.CATEGORY_FIELD_OF_CLASS),

        }

        for node_id in all_node_id_set:
            if node_id in finish_node_id_set:
                continue

            node_json = self.graph_data.get_node_info_dict(node_id=node_id)
            finish_node_id_set.add(node_id)

            valid = False
            for label in VALID_LABELS:
                if label in node_json[GraphData.DEFAULT_KEY_NODE_LABELS]:
                    valid = True
                    break

            if not valid:
                continue

            qualified_name = node_json[GraphData.DEFAULT_KEY_NODE_PROPERTIES]["qualified_name"]
            child_node_id = node_json[GraphData.DEFAULT_KEY_NODE_ID]

            parent_qualified_name = self.get_parent_name_for_api(qualified_name)

            if not parent_qualified_name or qualified_name == parent_qualified_name:
                continue

            parent_node_json = self.graph_data.find_one_node_by_property("qualified_name", parent_qualified_name)

            if parent_node_json is None:
                print("%r can't found" % parent_qualified_name)
                continue
            parent_node_id = parent_node_json[GraphData.DEFAULT_KEY_NODE_ID]

            self.graph_data.add_relation(child_node_id,
                                         CodeEntityRelationCategory.to_str(
                                             CodeEntityRelationCategory.RELATION_CATEGORY_BELONG_TO),
                                         parent_node_id)

        print(self.graph_data)
        self.graph_data.print_graph_info()
        print("end build belong to relation")

    def build_value_subclass_relation(self):
        """
        for all methods, add a abstract new node stand for the overloading method with different parameters.
        its qualified name is simple method name without parameters. eg. java.lang.Math.abs
        :return:
        """
        print("start build value subclass relation")
        print(self.graph_data)
        self.graph_data.print_graph_info()

        all_node_id_set = self.graph_data.get_node_ids()
        finish_node_id_set = set([])

        VALUE_CAT_STR = CodeEntityCategory.to_str(CodeEntityCategory.CATEGORY_VALUE)

        for node_id in all_node_id_set:
            if node_id in finish_node_id_set:
                continue

            node_json = self.graph_data.get_node_info_dict(node_id=node_id)
            finish_node_id_set.add(node_id)

            if VALUE_CAT_STR not in node_json[GraphData.DEFAULT_KEY_NODE_LABELS]:
                continue

            parameter_qualified_name = node_json[GraphData.DEFAULT_KEY_NODE_PROPERTIES]["qualified_name"]

            short_description = node_json[GraphData.DEFAULT_KEY_NODE_PROPERTIES]["short_description"]
            value_type = node_json[GraphData.DEFAULT_KEY_NODE_PROPERTIES]["type"]
            value_name = node_json[GraphData.DEFAULT_KEY_NODE_PROPERTIES]["value_name"]

            if not short_description:
                base_value_node_id = self.add_base_value_entity_node(value_type=value_type,
                                                                     value_name=value_name,
                                                                     short_description="",
                                                                     entity_category=CodeEntityCategory.CATEGORY_VALUE)
            else:
                base_value_node_id = node_json[GraphData.DEFAULT_KEY_NODE_ID]

            finish_node_id_set.add(base_value_node_id)

            candidate_nodes = self.graph_data.find_nodes_by_property(
                property_name="qualified_name", property_value=parameter_qualified_name)

            for candidate_node in candidate_nodes:
                team_method_node_id = candidate_node[GraphData.DEFAULT_KEY_NODE_ID]
                if team_method_node_id in finish_node_id_set:
                    continue
                finish_node_id_set.add(team_method_node_id)
                if team_method_node_id == base_value_node_id:
                    continue

                self.graph_data.add_relation(team_method_node_id,
                                             CodeEntityRelationCategory.to_str(
                                                 CodeEntityRelationCategory.RELATION_CATEGORY_SUBCLASS_OF),
                                             base_value_node_id)

            super_base_value_node_id = self.add_base_value_entity_node(value_type=value_type,
                                                                       value_name="<V>",
                                                                       short_description="",
                                                                       entity_category=CodeEntityCategory.CATEGORY_VALUE)
            finish_node_id_set.add(super_base_value_node_id)

            self.graph_data.add_relation(base_value_node_id,
                                         CodeEntityRelationCategory.to_str(
                                             CodeEntityRelationCategory.RELATION_CATEGORY_SUBCLASS_OF),
                                         super_base_value_node_id)
        print(self.graph_data)
        self.graph_data.print_graph_info()
        print("end build value subclass relation")

    def get_override_method_pairs(self, start_class_id, end_class_id):
        print("start try to locate override relation between %r - %r" % (start_class_id, end_class_id))

        start_class_node_json = self.graph_data.get_node_info_dict(start_class_id)
        end_class_node_json = self.graph_data.get_node_info_dict(end_class_id)

        start_class_name = start_class_node_json[GraphData.DEFAULT_KEY_NODE_PROPERTIES][
            GraphData.DEFAULT_KEY_PROPERTY_QUALIFIED_NAME]

        end_class_name = end_class_node_json[GraphData.DEFAULT_KEY_NODE_PROPERTIES][
            GraphData.DEFAULT_KEY_PROPERTY_QUALIFIED_NAME]

        in_relations = self.graph_data.get_all_in_relation_dict_list(start_class_id)
        print("try to locate override relation between %r - %r" % (start_class_name, end_class_name))

        belong_to_relation_str = CodeEntityRelationCategory.to_str(
            CodeEntityRelationCategory.RELATION_CATEGORY_BELONG_TO)

        override_method_pairs = []
        for (method_node_id, temp_r_type, _) in in_relations:
            if temp_r_type != belong_to_relation_str:
                continue
            method_node_json = self.graph_data.get_node_info_dict(method_node_id)
            if not method_node_json:
                continue

            if CodeEntityCategory.to_str(CodeEntityCategory.CATEGORY_CONSTRUCT_METHOD) in method_node_json[
                GraphData.DEFAULT_KEY_NODE_LABELS]:
                continue
            method_qualified_name = method_node_json[GraphData.DEFAULT_KEY_NODE_PROPERTIES][
                GraphData.DEFAULT_KEY_PROPERTY_QUALIFIED_NAME]
            if ".<init>" in method_qualified_name:
                continue
            if start_class_name not in method_qualified_name:
                print("error the method name %r don't contain class name %r" % (
                    method_qualified_name, start_class_name))
                continue

            new_method_qualified_name = method_qualified_name.replace(start_class_name, end_class_name)

            parent_method_node_json = self.graph_data.find_one_node_by_property(
                property_name=GraphData.DEFAULT_KEY_PROPERTY_QUALIFIED_NAME,
                property_value=new_method_qualified_name)

            if not parent_method_node_json:
                continue

            override_pair = (method_node_id, parent_method_node_json[GraphData.DEFAULT_KEY_NODE_ID])
            override_method_pairs.append(override_pair)
            print("found %r override %r" % (method_qualified_name, new_method_qualified_name))
        return override_method_pairs

    def build_override_relation(self):
        print("start build override relation")
        print(self.graph_data)
        self.graph_data.print_graph_info()

        VALID_LABELS = {
            CodeEntityCategory.to_str(CodeEntityCategory.CATEGORY_CLASS),
            CodeEntityCategory.to_str(CodeEntityCategory.CATEGORY_INTERFACE),
        }

        extends_relation_str = CodeEntityRelationCategory.to_str(
            CodeEntityRelationCategory.RELATION_CATEGORY_EXTENDS)

        extend_relation_list = self.graph_data.get_relation_list(extends_relation_str)

        for (start_class_id, relation_type, end_class_id) in extend_relation_list:

            override_method_pairs = self.get_override_method_pairs(start_class_id, end_class_id)

            for start_method_id, end_method_id in override_method_pairs:
                self.graph_data.add_relation(start_method_id,
                                             CodeEntityRelationCategory.to_str(
                                                 CodeEntityRelationCategory.RELATION_CATEGORY_METHOD_OVERRIDING),
                                             end_method_id)

        print(self.graph_data)
        self.graph_data.print_graph_info()
        print("start build override relation")

    def add_source_label(self, label):
        print("adding the source label %r to nodes in graph" % label)
        self.graph_data.add_label_to_all(label)

    def build(self):
        """
        get the graph after builder
        :return: the graph data build successfully
        """
        return self.graph_data
