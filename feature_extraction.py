# -*- coding: utf-8 -*-
import cv2

extractors = [("sift", cv2.SIFT()), ("surf", cv2.SURF())]
SUPPORTED_DESCRIPTORS = dict(extractors)

def extract_descriptors(image, descriptor='sift'):
    """ Given an image, it extracts keypoints and compute its descriptors using a selected technique"""
    if descriptor not in SUPPORTED_DESCRIPTORS:
        raise Exception("{0} descriptor not implemented".format(descriptor))
    keypoints, descriptors = SUPPORTED_DESCRIPTORS[descriptor].detectAndCompute(image, None)
    return descriptors

