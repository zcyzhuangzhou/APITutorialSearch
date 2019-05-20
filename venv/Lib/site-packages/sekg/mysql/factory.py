import json
import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


class MysqlSessionFactory:
    DB_URL_SCHEMA = "mysql+pymysql://{user}:{password}@{host}/{database}?charset=utf8"
    SERVER_URL_SCHEMA = "mysql+pymysql://{user}:{password}@{host}"

    def __init__(self, config_file_path):
        """
        init the mysql session by a config path.
        the config json file format example:
        [
            {
                "server_name": "LocalHostServer",
                "server_id": 1,
                "host": "10.141.221.87",
                "user": "root",
                "password": "root",
                "databases": [
                        {
                            "database_id": 1,
                            "database_name": "testsekg",
                            "description": "the test sekg se"
                        },
                        {
                            "database_id": 2,
                            "database_name": "test",
                            "description": "the test server"
                        }
                ],
            },
            ...
        ]
        :param config_file_path: the config file path
        """
        if not os.path.exists(config_file_path):
            raise IOError("MySQL config file not exist")
        if not os.path.isfile(config_file_path):
            raise IOError("MySQL config path is not file")
        if not config_file_path.endswith(".json"):
            raise IOError("MySQL config file is not json")

        self.config_file_path = config_file_path
        with open(self.config_file_path, 'r') as f:
            self.configs = json.load(f)
        ## todo add more json format check, raise exception when same name or same id config

    def create_mysql_engine(self, server_name=None, server_id=None, database=None, echo=False):
        """
        create a engine to mysql by sqlalchemy. one of server_name or server_id must given.
        database is None.create the engine to server, otherwise, create the engine to one database from one mysql server
        :param server_name: the server name in config file, can be used to find a unique mysql location
        :param database: the database name connect to. for mysql, one mysql in one IP server.
        one mysql could have many database with different name. if the database is None,create
        :param server_id: the server id in config file, can be used to find a unique mysql location
        :param echo: if true, all the sql executed will print to console
        :return: the engine object in sqlalchemy, None if create fail
        """
        config = self.get_config(server_name=server_name, server_id=server_id)
        if config is None:
            return None
        return self.__create_mysql_engine_by_config(config, database=database, echo=echo)

    def create_mysql_engine_by_server_name(self, server_name, database=None, echo=False):
        """
        create a engine to mysql by sqlalchemy. one of server_name or server_id must given.
        database is None.create the engine to server, otherwise, create the engine to one database from one mysql server
        :param server_name: the server name in config file, can be used to find a unique mysql location
        :param database: the database name connect to. for mysql, one mysql in one IP server.
        one mysql could have many database with different name. if the database is None,create
        :param echo: if true, all the sql executed will print to console
        :return: the engine object in sqlalchemy, None if create fail
        """
        return self.create_mysql_engine(server_name=server_name, database=database, echo=echo)

    def create_mysql_engine_by_server_id(self, server_id, database=None, echo=False):
        """
              create a engine to mysql by sqlalchemy. one of server_name or server_id must given.
        database is None.create the engine to server, otherwise, create the engine to one database from one mysql server
        :param database: the database name connect to. for mysql, one mysql in one IP server.
        one mysql could have many database with different name. if the database is None,create
        :param server_id: the server id in config file, can be used to find a unique mysql location
        :param echo: if true, all the sql executed will print to console
        :return: the engine object in sqlalchemy, None if create fail
        """

        return self.create_mysql_engine(server_id=server_id, database=database, echo=echo)

    def create_mysql_session(self, server_name=None, server_id=None, database=None, echo=False, autocommit=False):
        """
        create the session to one specify database from one server
        :param server_name: the server name in config file, can be used to find a unique mysql location
        :param database: the database name connect to. for mysql, one mysql in one IP server.
        one mysql could have many database with different name. if the database is None,create
        :param server_id: the server id in config file, can be used to find a unique mysql location
        :param echo: if true, all the sql executed will print to console
        :param autocommit: if True, all sql will be commit automately. If False,all write sql statement will be executed after session.commit() call.
        :return: the session object in sqlalchemy, None if create fail
        """
        if database is None:
            print("the database must be given when config")

            return None
        engine = self.create_mysql_engine(server_name=server_name, server_id=server_id, database=database, echo=echo)
        return self.create_mysql_session_from_engine(engine=engine, autocommit=autocommit, echo=echo)

    def create_mysql_session_by_server_name(self, server_name, database, echo=False, autocommit=False):
        """
        create the session to one specify database from one server
        :param server_name: the server name in config file, can be used to find a unique mysql location
        :param database: the database name connect to. for mysql, one mysql in one IP server.
        one mysql could have many database with different name. if the database is None,create
        :param echo: if true, all the sql executed will print to console
        :param autocommit: if True, all sql will be commit automately. If False,all write sql statement will be executed after session.commit() call.
        :return: the session object in sqlalchemy, None if create fail
        """
        return self.create_mysql_session(server_name=server_name, database=database, echo=echo, autocommit=autocommit)

    def create_mysql_session_by_server_id(self, server_id, database, echo=False, autocommit=False):
        """
        create the session to one specify database from one server
        :param database: the database name connect to. for mysql, one mysql in one IP server.
        one mysql could have many database with different name. if the database is None,create
        :param server_id: the server id in config file, can be used to find a unique mysql location
        :param echo: if true, all the sql executed will print to console
        :param autocommit: if True, all sql will be commit automately. If False,all write sql statement will be executed after session.commit() call.
        :return: the session object in sqlalchemy, None if create fail
        """
        return self.create_mysql_session(server_id=server_id, database=database, echo=echo, autocommit=autocommit)

    @staticmethod
    def create_mysql_session_from_engine(engine, autocommit, echo):
        """
        create the session to one specify database from one server
        :param engine: the engine use to create engine
        :param echo: if true, all the sql executed will print to console
        :param autocommit: if True, all sql will be commit automately. If False,all write sql statement will be executed after session.commit() call.
        :return: the session object in sqlalchemy, None if create fail
        """
        if engine is None:
            print("engine create fail for create session")
            return None
        Session = sessionmaker(bind=engine, autocommit=autocommit)
        session = Session()

        if echo:
            print("create new session by %r" % autocommit)
        return session

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

    def __in_database_config_list(self, database, database_config_list):
        for config in database_config_list:
            if config["database_name"] == database:
                return True
        return False

    def __create_mysql_engine_by_config(self, config, database, echo):
        if database is not None:

            if not self.__in_database_config_list(database, config['databases']):
                print("crate engine fail, the database name not in config %r", config)
                return None
            url = self.DB_URL_SCHEMA.format(
                host=config['host'],
                database=database,
                user=config['user'],
                password=config['password'])
        else:
            url = self.SERVER_URL_SCHEMA.format(
                host=config['host'],
                user=config['user'],
                password=config['password'])

        engine = create_engine(url, encoding='utf-8',
                               echo=echo)
        if echo:
            print("create engine by url={url}".format(url=url))
        return engine

    def __create_all_databases_by_config(self, config, echo=True):
        """
        create all databases in one server by config dict
        :param config: the config dict
        :param echo: if true, all the sql executed will print to console
        """
        engine = self.create_mysql_engine(server_name=config["server_name"], echo=echo)
        for config in config["databases"]:
            self.create_database_in_engine(engine=engine, database=config["database_name"])

    def create_database_in_engine(self, engine, database):
        """
        create a new database in engine, if exist not create
        :param engine: the engine connected to
        :param database: the new database name
        :return:
        """
        try:

            # create db
            engine.execute("CREATE DATABASE IF NOT EXISTS {database}".format(database=database))
        except Exception as e:
            print("create database {database} in one server by config fail".format(database=database))
            print(e)

    def create_databases_in_engine(self, engine, databases):
        """
        create a new database in engine, if exist not create
        :param engine: the engine connected to
        :param databases: the new database name list
        :return:
        """
        for database in databases:
            self.create_database_in_engine(engine=engine, database=database)

    def create_all_databases(self, server_name=None, server_id=None, echo=True):
        """
        create all databases in one server by server_name or by server_id
        :param server_name: the server name in config file, can be used to find a unique mysql location
        :param server_id: the server id in config file, can be used to find a unique mysql location
        :param echo: if true, all the sql executed will print to console
        """
        config = self.get_config(server_id=server_id, server_name=server_name)
        self.__create_all_databases_by_config(config=config, echo=echo)

    def create_database_in_server(self, server_name=None, server_id=None, echo=True):
        """
        create all databases in one server by server_name or by server_id
        :param server_name: the server name in config file, can be used to find a unique mysql location
        :param server_id: the server id in config file, can be used to find a unique mysql location
        :param echo: if true, all the sql executed will print to console
        """
        config = self.get_config(server_id=server_id, server_name=server_name)
        self.__create_all_databases_by_config(config=config, echo=echo)

    def get_config_by_server_name(self, server_name):
        for config in self.configs:
            if config["server_name"] == server_name:
                return config
        return None

    def get_config_by_server_id(self, server_id):
        for config in self.configs:
            if config["server_id"] == server_id:
                return config
        return None

    def get_config(self, server_name=None, server_id=None):
        """
        find the config by server_name or server_id
        :param server_name: the server name in config file, can be used to find a unique mysql location
        :param server_id: the server id in config file, can be used to find a unique mysql location
        :return:
        """
        config = None
        if server_name is None and server_id is None:
            print("server_name and server_id can't both None")
            return None
        if server_name:
            config = self.get_config_by_server_name(server_name)
        if server_id:
            config = self.get_config_by_server_id(server_id)
        if config is None:
            print("can't find config")
            return None
        return config
