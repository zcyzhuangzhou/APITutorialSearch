import traceback

from sqlalchemy import text
from sqlalchemy.orm import sessionmaker

from .factory import MysqlSessionFactory


class MySQLAccessor:
    """
        This class wrap some query operation for mysql table with wrapping session, first you init this MySQLAccessor with a session.
        The session object is obtained by MysqlSessionFactory, it is from sqlalchemy library, connecting to one database
        in one mysql server, than you can use this accessor to do some query.
        You can extends this accessor to add more custom function.
    """

    def __init__(self, engine, autocommit=False, echo=True):
        self.engine = engine
        self.session = MysqlSessionFactory.create_mysql_session_from_engine(engine=engine, autocommit=autocommit,
                                                                            echo=echo)

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

    def is_connect(self):
        """
        check if the session valid
        :return: True, valid; False, not valid
        """
        if self.session is None:
            return False
        else:
            return True

    def delete_all(self, model_class):
        """
        delete all item in one table
        :param model_class: the model class extend Base, "Base = declarative_base()", it has a __tablename__ property
        to find the table
        :return:
        """
        try:
            self.session.query(model_class).delete()
            self.session.commit()
        except Exception as error:
            traceback.print_exc()
            return

    def create_orm_tables(self, SqlachemyORMBaseClass):
        """
        create all table, must given a SqlachemyORMBaseClass
        :param SqlachemyORMBaseClass: SqlachemyORMBaseClass that every table model extended
        :return:
        """
        # create the table
        SqlachemyORMBaseClass.metadata.create_all(bind=self.engine)

    def drop_orm_tables(self, SqlachemyORMBaseClass):
        """
                create all table, must given a SqlachemyORMBaseClass
                :param SqlachemyORMBaseClass: SqlachemyORMBaseClass that every table model extended
                :return:
        """

        # delete all table
        SqlachemyORMBaseClass.metadata.drop_all(bind=self.engine)

    def create_table_by_metadata(self, metadata):
        """
        create a table by a metadata
        example: create table by metadata
        metadata = MetaData(engine)

        user = Table('user', metadata,
        Column('id', Integer, primary_key = True),
        Column('name', String(20)),
        Column('fullname', String(40)))
        address = Table('address', metadata,
            Column('id', Integer, primary_key = True),
            Column('user_id', None, ForeignKey('user.id')),
            Column('email', String(60), nullable = False),
        )
        metadata.create_all(engine)

        :param metadata:
        :return:
        """
        metadata.create_all(engine=self.engine)

    def get_by_primary_key(self, model_class, primary_property, primary_property_value):
        """
        get one Model class object by query the Mysql, the primary property value must be the primary_property_value.

        example: model_class=APIEntity,primary_property="id",primary_property_value="3", get the apiEntity where "id"=3
               :param model_class: the model class extend Base, "Base = declarative_base()", it has a __tablename__ property
               to find the table
               :param primary_property: the primary_property in model_class to find a unique object. etc. APIEntity.id
                :param primary_property_value: the primary_property_value that primary_property is

               :return:
        """

        try:
            return self.session.query(model_class).filter(primary_property == primary_property_value).first()
        except Exception:
            traceback.print_exc()
            return None

    def add_index(self, index_name, table_name, column_name):
        """
        add index to a table in one column
        :param index_name: the index name
        :param table_name: the table name
        :param column_name: the column name
        :return:
        """
        # todo: need test
        try:
            conn = self.engine.connect()
            text_sql = 'alter table {table_name} add index {index_name}(column_name)'.format(
                table_name=table_name,
                index_name=index_name,
                column_name=column_name)

            s = text(text_sql)

            conn.execute(s)
            conn.close()
        except:
            traceback.print_exc()

    def add_multi_index(self, index_name, table_name, *column_name_list):
        """
        add index to a table in one column
        :param index_name: the index name
        :param table_name: the table name
        :param column_name_list: the column name list
        :return:
        """
        # todo: need test
        try:
            conn = self.engine.connect()
            text_sql = 'alter table {table_name} add index {index_name}(column_name)'.format(
                table_name=table_name,
                index_name=index_name,
                column_name=",".join(column_name_list))

            s = text(text_sql)

            conn.execute(s)
            conn.close()
        except:
            traceback.print_exc()

    def delete_multi_index(self, index_name, table_name, *column_name_list):
        """
        delete index to a table in one column
        :param index_name:  the index name
        :param table_name: the table name
        :param column_name_list: the column name list
        :return:
        """
        # todo: need test

        try:
            conn = self.engine.connect()
            text_sql = 'alter table {table_name} drop index {index_name}(column_name)'.format(
                table_name=table_name,
                index_name=index_name,
                column_name=",".join(column_name_list))

            s = text(text_sql)

            conn.execute(s)
            conn.close()
        except:
            traceback.print_exc()

    def query_all(self, model_class):
        """
        query all item in one table
        :param model_class: the model class extend Base, "Base = declarative_base()", it has a __tablename__ property
        to find the table
        :return: the all result
        """
        try:
            self.session.query(model_class).all()
        except Exception as error:
            traceback.print_exc()
            return []

    def create_without_duplicate(self, autocommit=True):
        # todo: fix this method
        pass
