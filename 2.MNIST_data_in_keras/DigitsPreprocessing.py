# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 17:25:11 2024

@author: Javier Andrango
"""

import cv2
import numpy as np
import os 
def process_new_images(path):
    '''
    Parameters
    ----------
    path : String
        Images folder name.

    Returns
    -------
    images : Array
        Array of matrices(images).
    '''
    images = []
    for filename in os.listdir(path):
        img = cv2.imread(os.path.join(path,filename))
        if img is not None:
            # resize image
            img = cv2.resize(img,(28,28))
            # GRAY image
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # filter
            thresh_val, thresh_matrix = cv2.threshold(img, 120, 1,0)
            thresh_matrix_inv = 1 - thresh_matrix
            img = np.multiply((255-img),thresh_matrix_inv)
            # clarify the image 25%
            white_filter = np.dot(np.ones((28,28)),1.25)
            img = np.multiply(img,white_filter)
            img = img.astype(int)
            img[img>255]=255
        # save processed image
        images.append(img)
    return images


def labels_from_images(path):
    '''
    Parameters
    ----------
    path : String
        Images folder name.

    Returns
    -------
    labels : Array
        Array of labels (images labels)

    '''
    labels = []
    for filename in os.listdir(path):
        # save the label from the image name
        labels.append(int(filename[0]))
    return labels


def custom_dataset(path):
    images = process_new_images(path)
    labels = labels_from_images(path)
    return images, labels
