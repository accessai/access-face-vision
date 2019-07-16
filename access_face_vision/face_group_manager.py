import os
import logging
from glob import glob
from threading import Lock
import uuid
from time import time

import numpy as np
from numpy import savez_compressed

from access_face_vision.db.mongo_manager import MongoManager
from access_face_vision.exceptions import AccessException

logger = logging.getLogger(__name__)


class FaceGroup(object):

    def __init__(self, faceIds, embeddings, labels):
        self.faceIds = faceIds
        self.embeddings = embeddings
        self.labels = labels


class FaceGroupManager(object):

    def __init__(self):
        self.io_lock = Lock()

    def generate_face_id(self, label):
        return "{}-{}-{}".format(label[:4], str(uuid.uuid4()), str(int(time()))[-8:])


class FaceGroupMongoManager(FaceGroupManager):

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


class FaceGroupLocalManager(FaceGroupManager):

    def __init__(self, cmd_args):
        super(FaceGroupLocalManager, self).__init__()
        self.dir_name = cmd_args.face_group_dir
        self.face_group_store = {}
        self.fill_face_group_store()

    def create_face_group(self, fg_name):
        self._save_face_group(fg_name, FaceGroup(np.array([]), np.array([]), np.array([])), overwrite=False)

    def append_to_face_group(self, fg_name, embedding, label):
        faceId = self.generate_face_id(label)
        fg = self._get_face_group(fg_name)
        if len(fg.embeddings)==0:
            fg.embeddings = np.empty((0,embedding.size), dtype=np.float32)
        fg.embeddings = np.append(fg.embeddings, np.array([embedding]), axis=0)
        fg.labels = np.append(fg.labels, np.array([label]), axis=0)
        fg.faceIds = np.append(fg.faceIds, np.array([faceId]), axis=0)

        self._save_face_group(fg_name, fg)
        return faceId

    def get_face_group(self, fg_name):
        return self._get_face_group(fg_name)

    def delete_from_face_group(self, face_id, fg_name):

        fg = self._get_face_group(fg_name)

        delete_index = np.where(fg.faceIds == face_id)

        if len(delete_index) > 0:
            delete_index = delete_index[0]
            faceIds = np.delete(fg.faceIds, delete_index, axis=0)
            embeddings = np.delete(fg.embeddings, delete_index, axis=0)
            labels = np.delete(fg.labels, delete_index, axis=0)

            fg.faceIds = faceIds
            fg.embeddings = embeddings
            fg.labels = labels

            self._save_face_group(fg_name, fg)
            return fg

        else:
            raise AccessException('faceId not found: {}'.format(face_id))

    def delete_face_group(self, fg_name):
        self._delete_face_group(fg_name)

    def fill_face_group_store(self):
        face_groups = glob(os.path.join(self.dir_name, "**/*.npz"))[:5]
        for fg in face_groups:
            fg_name = os.path.basename(fg).split(".")[0]
            self._get_face_group(fg_name)  # updates self.face_group_store internally

    def get_file_path(self, fg_name):
        return os.path.join(self.dir_name, fg_name + '.npz')

    def _save_face_group(self, fg_name, fg, overwrite=True):

        file_path = self.get_file_path(fg_name)
        if os.path.exists(file_path) and overwrite is False:
            raise AccessException("FaceGroup {} already exists.".format(fg_name), error_code=409)

        with self.io_lock:
            savez_compressed(file_path, faceIds=fg.faceIds, embeddings=fg.embeddings,
                             labels=fg.labels)
            self.face_group_store[fg_name] = fg

    def _get_face_group(self, fg_name):
        with self.io_lock:
            fg = self.face_group_store.get(fg_name)
            if fg is not None:
                return self.face_group_store.get(fg_name)
            else:
                file_path = self.get_file_path(fg_name)

                try:
                    fg = np.load(file_path)
                except IOError as ioe:
                    raise AccessException(str(ioe))
                except ValueError as ve:
                    raise AccessException(str(ve))

                fg = FaceGroup(fg['faceIds'], fg['embeddings'], fg['labels'])
                self.face_group_store[fg_name] = fg
                return fg

    def _delete_face_group(self, fg_name):
        with self.io_lock:
            file_path = self.get_file_path(fg_name)
            try:
                del self.face_group_store[fg_name]
                os.remove(file_path)
            except Exception as ex:
                raise AccessException('Could not delete {}. Error {}'.format(file_path, str(ex)))
