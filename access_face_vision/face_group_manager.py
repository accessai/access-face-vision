import os
import logging

import numpy as np
from numpy import savez_compressed

from access_face_vision.db.mongo_manager import MongoManager
from access_face_vision.exceptions import AccessException

logger = logging.getLogger(__name__)


class FaceGroup(object):
    def __init__(self):
        pass

    def create_face_group(self, face_group_name):
        pass

    def append_to_face_group(self, face, face_group_name):
        pass

    def delete_from_face_group(self, face_id, face_group_name):
        pass

    def delete_face_group(self, face_group_name):
        pass


class FaceGroupMongoManager(FaceGroup):

    def __init__(self, mongo_connect_str):
        super(FaceGroupMongoManager, self).__init__()
        self.connection_str = mongo_connect_str
        self.mongo_manager = MongoManager()
        self.mongo_client = self.mongo_manager.get_client()
        self.mongo_db = self.mongo_manager.get_db(self.mongo_client)

    def _get_collection(self, collection_name):
        return self.mongo_manager.get_collection(self.mongo_db, collection_name)

    def create_face_group(self, face_group_name):
        return self._get_collection(face_group_name)

    def append_to_face_group(self, face, face_group_name):
        collection = self._get_collection(face_group_name)
        return self.mongo_manager.insert_doc(face, collection)

    def delete_from_face_group(self, face_id, face_group_name):
        collection = self._get_collection(face_group_name)
        return self.mongo_manager.delete_records({'faceId': face_id}, collection)

    def delete_face_group(self, face_group_name):
        collection = self._get_collection(face_group_name)
        self.mongo_manager.delete_collection(collection)


class FaceGroupLocalManager(FaceGroup):

    def __init__(self, dir_name):
        super(FaceGroupLocalManager, self).__init__()
        self.dir_name = dir_name

    def _get_file_path(self, face_group_name):
        return os.path.join(self.dir_name, face_group_name + '.npz')

    def _save_face_group(self, file_path, faceIds, embeddings, labels):
        savez_compressed(file_path, faceIds=faceIds, embeddings=embeddings, labels=labels)

    def _load_face_group(self, file_path):
        face_group = np.load(file_path)
        faceIds = face_group['faceIds']
        embeddings = face_group['embeddings']
        labels = face_group['labels']

        return faceIds, embeddings, labels

    def create_face_group(self, face_group_name):
        file_path = self._get_file_path(face_group_name)
        savez_compressed(file_path, faceIds=np.array([]), embeddings=np.array([]), labels=np.array([]))

    def append_to_face_group(self, face, face_group_name):
        file_path = self._get_file_path(face_group_name)
        faceIds, embeddings, labels = self._load_face_group(file_path)

        embeddings.append(face['embedding'])
        labels.append(face['label'])
        faceIds.append(face['faceId'])

        self._save_face_group(file_path, faceIds, embeddings, labels)
        return faceIds, embeddings, labels

    def delete_from_face_group(self, face_id, face_group_name):
        file_path = self._get_file_path(face_group_name)
        faceIds, embeddings, labels = self._load_face_group(file_path)

        delete_index = np.where(faceIds==face_id)

        if len(delete_index) > 0:
            delete_index = delete_index[0]
            faceIds = np.delete(faceIds, delete_index, axis=0)
            embeddings = np.delete(embeddings, delete_index, axis=0)
            labels = np.delete(labels, delete_index, axis=0)

            self._save_face_group(file_path, faceIds, embeddings, labels)
            return faceIds, embeddings, labels

        else:
            raise AccessException('faceId not found: {}'.format(face_id))

    def delete_face_group(self, face_group_name):
        file_path = self._get_file_path(face_group_name)
        try:
            os.remove(file_path)
        except:
            raise AccessException('Could not delete {}'.format(file_path))