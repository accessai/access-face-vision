import json

from keras.preprocessing.image import  ImageDataGenerator
from keras.models import load_model
from numpy import savez_compressed, load
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
import pickle
from sklearn.preprocessing import Normalizer

class Trainer(object):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data_generator = ImageDataGenerator()
        self.model = load_model('D:\models\\facenet_keras.h5')

    def generate_face_embeddings(self):
        generator = self.data_generator.flow_from_directory('D:\data\PINS',
                                                            target_size=(160,160))

        embeddings = []
        labels = []
        count=0
        for data_batch, labels_batch in generator:
            embeds = self.model.predict(data_batch)
            # mean, std = embeds.mean(axis=1).reshape(-1,1), embeds.std(axis=1).reshape(-1,1)
            # embeds = (embeds - mean) / std
            embeddings.extend(embeds)
            labels.extend(np.argmax(labels_batch, axis=1))
            count += 32
            print(count)
            if count>10776:
                break

        embeddings = np.array(embeddings)
        labels = np.array(labels)
        with open('class.json', 'w') as f:
            json.dump(generator.class_indices, f)
        X_train, X_, Y_train, Y_ = train_test_split(embeddings, labels, test_size=0.4, stratify=labels)
        savez_compressed('train.npz', embeddings=X_train, labels=Y_train)
        X_test, X_val, Y_test, Y_val = train_test_split(X_, Y_, test_size=0.5, stratify=Y_)
        savez_compressed('test.npz', embeddings=X_test, labels=Y_test)
        savez_compressed('val.npz', embeddings=X_val, labels=Y_val)

    def face_classifier(self):
        normalizer = Normalizer(norm='l2')
        svc_model = SVC(kernel='linear')
        data = load('train.npz')
        X_train, Y_train = data['embeddings'], data['labels']

        data = load('test.npz')
        X_test, Y_test = data['embeddings'], data['labels']

        data = load('val.npz')
        X_val, Y_val = data['embeddings'], data['labels']

        svc_model.fit(X_train, Y_train)
        Y_hat = svc_model.predict(X_test)

        print('Test accuracy: ', accuracy_score(Y_test,Y_hat))
        print('Test f1: ', f1_score(Y_test,Y_hat, average='weighted'))
        Y_hat = svc_model.predict(X_val)
        print('Val accuracy: ', accuracy_score(Y_val, Y_hat))
        print('Val f1: ', f1_score(Y_val, Y_hat, average='weighted'))

        with open('classifier.pkl', 'wb') as f:
            pickle.dump(svc_model, f)


if __name__ == '__main__':
    trainer = Trainer('D:\data\PINS')
    # trainer.face_classifier()
    trainer.generate_face_embeddings()