#importing libraries 
import numpy as np
import pandas as pd

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import cv2

from tqdm import tqdm_notebook as tqdm
from functools import partial
import scipy as sp

import random
import time
import sys
import os
import yaml

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import metrics
from sklearn.metrics import confusion_matrix
import torch
import torchvision

import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms, models, datasets
from torchvision.models.feature_extraction import create_feature_extractor
from torch.utils.data import Dataset
from torch.autograd import Variable

import warnings
warnings.filterwarnings('ignore')

from utils.preprocessing import *    
    
class load_data:
    
    def __init__(self, train_labels_path, test_labels_path, train_image_path, test_image_path, columns, 
              itype = '.jpg', batch_size = 16, shuffle=True, do_random_crop = False, device = 'cpu'):
        self.train_labels_path = train_labels_path
        self.test_labels_path = test_labels_path
        self.train_image_path = train_image_path
        self.test_image_path = test_image_path
        self.columns = columns
        self.train_df, self.test_df, self.valid_df = self.import_data()
        self.itype = itype
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.do_random_crop = do_random_crop
        self.device = device
        
        #GPU CHECK
        train_on_gpu = torch.cuda.is_available()
        if not train_on_gpu:
            device = torch.device('cpu')
        else:
            device = torch.device('cuda:0')
        
        # train transformations
        self.train_trans = transforms.Compose([transforms.ToPILImage(),
                                        transforms.RandomRotation((-360, 360)),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomVerticalFlip(),
                                        transforms.ToTensor()
                                        ])

        # validation transformations
        self.valid_trans = transforms.Compose([transforms.ToPILImage(),
                                        transforms.ToTensor(),
                                        ])

        # test transformations
        self.test_trans = self.valid_trans
        

        
    def import_data(self):
        train_df = pd.read_csv(self.train_labels_path)
        test_df = pd.read_csv(self.test_labels_path)
        n = int(len(test_df)/2)
        test_df = test_df.iloc[:n, 0:2]
        valid_df = test_df.iloc[n:, 0:2]
        return train_df, test_df, valid_df
        
    def create_loader(self):
        
        train_dataset = EyeData(data = self.train_df.iloc[:, 0:2], 
                                directory  = self.train_image_path,
                                transform  = self.train_trans,
                                columns = self.columns,
                                itype = self.itype)

        train_loader = torch.utils.data.DataLoader(dataset     = train_dataset, 
                                                    batch_size  = self.batch_size, 
                                                    shuffle     = self.shuffle, 
                                                    num_workers = 4)
        
        test_dataset = EyeData(data = self.test_df.iloc[:, 0:2], 
                            directory  = self.test_image_path,
                            transform  = self.test_trans,
                            columns = self.columns,
                            itype = self.itype,
                            do_random_crop = False)

        test_loader = torch.utils.data.DataLoader(dataset     = test_dataset, 
                                                    batch_size  = int(self.batch_size/4), 
                                                    shuffle     = False, 
                                                    num_workers = 4)
        
        valid_dataset = EyeData(data = self.valid_df.iloc[:, 0:2], 
                            directory  = self.test_image_path,
                            transform  = self.valid_trans,
                            columns = self.columns,
                            itype = self.itype,
                            do_random_crop = False)

        valid_loader = torch.utils.data.DataLoader(dataset     = valid_dataset, 
                                                    batch_size  = int(self.batch_size/4), 
                                                    shuffle     = False, 
                                                    num_workers = 4)
        
        return train_loader, test_loader, valid_loader
    
    