# -*- coding: utf-8 -*-
import cv2
import os,sys
import numpy as np
from scipy.cluster import vq

extractors = [("sift", cv2.SIFT()), ("surf", cv2.SURF())]
SUPPORTED_DESCRIPTORS = dict(extractors)

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
    data = np.array(samples, dtype=np.float32)
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

