# -*- coding: utf-8 -*-
from lshash import LSHash


class Searcher:

    _DIST_FUNCTIONS = ["hamming", "euclidean", "true_euclidean", "centred_euclidean", "cosine", "l1norm"]
    index = None

    def __init__(self, dataset):
        self.create_index(dataset)

    def create_index(self, items, hash_size=6):
        input_dim = len(items.values()[0])
        self.index = LSHash(hash_size, input_dim)
        for key in items:
            self.index.index(items[key], extra_data=key)
        return True

    def query(self, query_item, num_results=10, distance_function='cosine'):
        if distance_function not in self._DIST_FUNCTIONS:
            raise Exception("{0} not supported".format(distance_function))
        results = self.index.query(query_item, num_results=num_results, distance_func=distance_function)
        return self.parse_results(results)

    def parse_results(self, results):
        return {x[0][1]:x[1] for x in results}

