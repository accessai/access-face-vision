import os
import logging

logger = logging.getLogger('MONGODB')

import pymongo


class MongoManager(object):
    def __init__(self):
        self.mongo_connect_str = os.getenv('MONGO_CONNECT_STR', None)
        self.mongo_db_name = os.getenv('MONGO_DB_NAME', None)
        self.mongo_user = os.getenv('MONGO_USER', None)
        self.mongo_pwd = os.getenv('MONGO_PWD', None)

    def get_client(self):
        try:
            return pymongo.MongoClient(self.mongo_connect_str, username=self.mongo_user, password=self.mongo_pwd)
        except Exception as ex:
            logger.error("Error connecting to DB. {}".format(ex))
            return None

    def get_db(self, client):
        return client[self.mongo_db_name]

    def get_collection(self, db, collection_name):
        return db[collection_name]

    def insert_doc(self, doc, collection):
        return collection.insert_one(doc).inserted_id

    def get_doc(self,query_doc, collection):
        return collection.find(query_doc, max_time_ms=500)

    def delete_records(self, doc, collection):
        return collection.remove(doc)

    def delete_collection(self, collection):
        return collection.drop()