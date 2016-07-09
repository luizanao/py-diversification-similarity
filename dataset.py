# -*- coding: utf-8 -*-
import cv2, os
from collections import namedtuple

Dataset = namedtuple("Dataset", "name, path, data")

def load_dataset(path):
    dataset = []
    for subdir, dirs, files in os.walk(path):
        for file in files:
            abs_path = os.path.join(subdir, file)
            img_data = cv2.imread(abs_path)
            dataset.append(Dataset(name=file, path=abs_path, data=img_data))
    return dataset

