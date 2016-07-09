# -*- coding: utf-8 -*-
import cv2
import os,sys
import Image
import numpy as np
from scipy.cluster import vq

extractors = [("sift", cv2.SIFT()), ("surf", cv2.SURF())]
SUPPORTED_DESCRIPTORS = dict(extractors)

def io_loader(query_object,dataset):
    '''
    dataset: string (path)(multiple obj)
    query_object: string (path)(single obj)
    Load entire datase and search point, and call features extraction 
    
    could not be tested - TODO test and refactor
    '''
    if not (query_object and dataset):
        return     

    query_object_features = extract_descriptors(Image.open(query_object), descriptor='sift')
    datase_features = []
    for subdir, dirs, files in os.walk(dataset):
        for file in files:
            img_path = os.path.join(subdir, file)
            datase_features.append(extract_descriptors(Image.open(img_path), descriptor='sift'))



def extract_descriptors(image, descriptor='sift'):
    """ Given an image, it extracts keypoints and compute its descriptors using a selected technique"""
    if descriptor not in SUPPORTED_DESCRIPTORS:
        raise Exception("{0} descriptor not implemented".format(descriptor))
    keypoints, descriptors = SUPPORTED_DESCRIPTORS[descriptor].detectAndCompute(image, None)
    return descriptors


def create_codebook(samples, size=300):
    """ Given a list of samples and the number of clusters, it applies a clusterization over all
    descriptors and generates a visual dictionary named codebook"""
    attempts = 1
    data = np.array(samples)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)
    ret, labels, centroids = cv2.kmeans(data, size, criteria, attempts, cv2.KMEANS_PP_CENTERS)
    return centroids


def extract_features(codebook, descriptors):
    """Converts an array of descriptors into an array of features using a pre-calculated codebook"""
    features = np.zeros(len(codebook))
    indexes, dist = vq.vq(descriptors, codebook)
    for idx in indexes:
        features[idx] += 1
    return features

if __name__ == '__main__':
    _path = '/home/luizfelipe/Downloads/zip1'
    io_loader(_path + '/all_souls_000152.jpg', _path)