# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal.
"""
import cv2
import datetime as dt
import h5py
import matplotlib.pyplot as plt
import matplotlib.pylab as plb
import numpy as np
import os
import pandas as pd
from glob import glob

def proc_images():
    """
    Saves compressed, resized images as HDF5 datsets
    Returns
        data.h5, where each dataset is an image or class label
        e.g. X23,y23 = image and corresponding class label
    """
    dataset_name = "waifus"
    dataset_type = "train"
    dataset_fullname = dataset_type + '_' + dataset_name
    start = dt.datetime.now()
    
    # ../datasets/
    PATH = os.path.abspath(os.path.join('..', 'datasets'))
    # ../datasets/raw/images/
    SOURCE_IMAGES = os.path.join(PATH, "raw", dataset_type)
    # ../datasets/raw/images/*.png
    images = glob(os.path.join(SOURCE_IMAGES, "*.png"))
    images+= glob(os.path.join(SOURCE_IMAGES, "*.jpeg"))
    
    # Load labels
    labels = pd.read_csv(os.path.join(SOURCE_IMAGES, dataset_fullname + '.csv'))
    
    # Size of data
    NUM_IMAGES = len(images)
    HEIGHT = 64
    WIDTH = 64
    CHANNELS = 3
    SHAPE = (HEIGHT, WIDTH, CHANNELS)
    
    new_images = []
    print(len(labels["is waifu"].values))
    
    with h5py.File(dataset_fullname + '.h5', 'w') as hf:                
        for i,img in enumerate(images):            
            # Images
            image = cv2.imread(img)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (WIDTH,HEIGHT))            
            new_images.append(image)
            end=dt.datetime.now()
            print("\r", i, ": ", (end-start).seconds, "seconds", end="")

        Xset = hf.create_dataset(
                    name='train_set_x',
                    data=new_images,
                    shape=(NUM_IMAGES, HEIGHT, WIDTH, CHANNELS),
                    maxshape=(None, HEIGHT, WIDTH, None),
                    compression="gzip",
                    compression_opts=9)
        
        yset = hf.create_dataset(
                name='train_set_y',
                data=labels["is waifu"].values,
                #data=np.random.randint(2, size=NUM_IMAGES),
                shape=(NUM_IMAGES,),
                maxshape=(None,),
                compression="gzip",
                compression_opts=9)
proc_images()
