import json
import os

from py2neo import Graph


class GraphInstanceFactory:

    def __init__(self, config_file_path):
        """
        init the graph factory by a config path.
        the config json file format example:
        [
            {
                "server_name": "LocalHostServer",
                "server_id": 1,
                "host": "localhost",
                "user": "neo4j",
                "password": "123456",
                "http_port": 7474,
                "https_port": 7473,
                "bolt_port": 7687
            },
            ...
        ]
        :param config_file_path: the config file path
        """
        if not os.path.exists(config_file_path):
            raise IOError("Neo4j config file not exist")
        if not os.path.isfile(config_file_path):
            raise IOError("Neo4j config path is not file")
        if not config_file_path.endswith(".json"):
            raise IOError("Neo4j config file is not json")

        self.config_file_path = config_file_path
        with open(self.config_file_path, 'r') as f:
            self.configs = json.load(f)
        ## todo add more json format check,raise exception when same name or same id config

    def create_py2neo_graph_by_server_name(self, server_name):
        """
        :param server_name: the server name in config file, can be used to find a unique neo4j graph instance location
        :return: the Graph object in py2neo, None if create fail
        """

        for config in self.configs:
            if config["server_name"] == server_name:
                return self.__create_py2neo_graph_by_config(config)
        return None

    def create_py2neo_graph_by_server_id(self, server_id):
        """
        :param server_id: the server id in config file, can be used to find a unique neo4j graph instance location
        :return: the Graph object in py2neo, None if create fail
        """
        for config in self.configs:
            if config["server_id"] == server_id:
                return self.__create_py2neo_graph_by_config(config)
        return None

    def get_configs(self):
        """
        get the config server list
        :return: a list of config
        """
        return self.configs

    def get_config_file_path(self):
        """
        get the config file path
        :return: a string for config file path
        """
        return self.config_file_path

    def __create_py2neo_graph_by_config(self, config):
        return Graph(host=config['host'],
                     port=config['bolt_port'],
                     scheme="bolt",
                     user=config['user'],
                     password=config['password'])
