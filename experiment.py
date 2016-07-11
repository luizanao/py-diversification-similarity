# -*- coding: utf-8 -*-
import feature_extraction
import dataset
import pickle
import numpy as np
from searcher import Searcher

if __name__ == '__main__':
    # TODO: load experiment using params
    train_path = ""
    test_path = ""
    # loads images from given paths
    train = dataset.load_dataset(train_path)
    test = dataset.load_dataset(test_path)

    # extracts descriptors for train and test sets
    train_descriptors = {item.path:feature_extraction.extract_descriptors(item.data) for item in train}
    test_descriptors = {item.path:feature_extraction.extract_descriptors(item.data) for item in test}

    # creates codebook (default size=300) based on train samples
    codebook = feature_extraction.create_codebook(np.concatenate(train_descriptors.values()))

    # generate feature vectors for train and test based on previously calculated codebook
    train_features = {key:feature_extraction.extract_features(codebook, train_descriptors[key]) for key in train_descriptors}
    test_features = {key:feature_extraction.extract_features(codebook, test_descriptors[key]) for key in test_descriptors}

    # TODO: create a similarity matrix using all features

    # persists features, codebook and similarity matrix
    pickle.dump(train_features, open("train_features.pk", "wb"))
    pickle.dump(test_features, open("test_features.pk", "wb"))
    pickle.dump(codebook, open("codebook.pk", "wb"))

    # creates index using LSHash and train set
    searcher = Searcher(train_features)

    # persists index for future use
    pickle.dump(searcher, open("searcher.pk", "wb"))
